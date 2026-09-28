// The branch and bound of SMS++ on a weighted MaxSAT instance (a SATBlock,
// whose SATSolver runs OLL within a budget of calls of the SAT solver in
// each node), exposed to Python as an environment: at each node to branch
// on, the agent is given the graph of its residual formula [see
// SATResidualGraph in SATSolver.h] and chooses the variable to fix and the
// value of the first child, as Graph-Q-SAT chooses a decision of MiniSat.
// The enumeration is depth-first, on a stack of fixings and their undos,
// with the same primitives BranchAndXSolver uses (compute(), branch(),
// apply()), so that a policy learned here works in it through the
// GQSATBranchRule of SATBlockML.

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SATBlock.h"
#include "SATSolver.h"

namespace py = pybind11;
using namespace SMSpp_di_unipi_it;

class BnBEnv
{
 public:

  BnBEnv( const std::string & solver , int max_iter , int features ,
          double penalty )
   : f_solver( solver ) , f_max_iter( max_iter ) , f_features( features ) ,
     f_penalty( penalty ) {}

  ~BnBEnv() { clear(); }

  // loads the instance and goes to the first node to branch on: returns
  // (state, done)
  py::tuple reset( const std::string & file ) {
   clear();
   std::ifstream in( file );
   if( ! in )
    throw( std::invalid_argument( "BnBEnv::reset: cannot open " + file ) );
   f_sat = std::make_unique< SATBlock >();
   f_sat->load( in );
   f_sat->generate_abstract_variables();
   f_s = dynamic_cast< SATSolver * >( Solver::new_Solver( f_solver ) );
   if( ! f_s )
    throw( std::invalid_argument( "BnBEnv::reset: " + f_solver +
                                  " is not a SATSolver" ) );
   f_s->set_par( SATSolver::intMaxSAT , 1 );
   f_s->set_par( Solver::intMaxIter , f_max_iter );
   f_sat->register_Solver( f_s );
   f_incumbent = Inf< double >();
   f_lb = - Inf< double >();
   f_nodes = 0;
   f_done = ! advance( true );
   return( py::make_tuple( state() , f_done ) );
   }

  // fixes the variable of action / 2 of the current graph to true if the
  // action is even, false if odd, as the first child (the other one is
  // explored after it); a negative action leaves the choice to the rule of
  // the cores; returns (state, reward, done), the reward being - penalty
  // per node evaluated
  py::tuple step( long action ) {
   if( f_done )
    throw( std::logic_error( "BnBEnv::step: the enumeration is over" ) );
   const long before = f_nodes;
   std::vector< Change * > children;
   if( action < 0 )
    children = f_s->branch();
   else {
    if( action >= 2 * f_g.n_var )
     throw( std::invalid_argument( "BnBEnv::step: action " +
                                   std::to_string( action ) + " out of " +
                                   std::to_string( 2 * f_g.n_var ) ) );
    children = make_children( f_g.var[ action / 2 ] ,
                              action % 2 == 0 ? 1 : 0 );
    }
   push( children );
   f_done = ! advance( false );
   return( py::make_tuple( state() ,
                           - f_penalty * double( f_nodes - before ) ,
                           f_done ) );
   }

  double incumbent( void ) const { return( f_incumbent ); }
  double root_lb( void ) const { return( f_lb ); }
  long nodes( void ) const { return( f_nodes ); }
  long n_var( void ) const { return( f_g.n_var ); }
  unsigned int n_col( void ) const {
   return( SATResidualGraph::n_features( f_features ) );
   }

 private:

  // a level of the stack: the undo of the fixing that brought there, and
  // the other child, still to explore (nullptr once it is taken)
  struct Frame {
   Change * undo;
   Change * sibling;
   };

  static std::vector< Change * > make_children( unsigned int var ,
                                                double first ) {
   return( std::vector< Change * >{
             new SATBlockChange( SATBlockChange::eFixX ,
                                 Block::Subset{ var } ,
                                 std::vector< double >{ first } ) ,
             new SATBlockChange( SATBlockChange::eFixX ,
                                 Block::Subset{ var } ,
                                 std::vector< double >{ 1 - first } ) } );
   }

  // goes down into the first of the children, keeping the second
  void push( std::vector< Change * > & children ) {
   auto undo = f_s->apply( children[ 0 ] , true );
   delete children[ 0 ];
   f_stack.push_back( { undo , children[ 1 ] } );
   }

  // evaluates the current node: true if it has to be branched on
  bool evaluate( bool root ) {
   ++f_nodes;
   const int status = f_s->compute();
   if( status == Solver::kInfeasible )
    return( false );
   if( status != Solver::kOK )
    throw( std::runtime_error( "BnBEnv: the SATSolver returned " +
                               std::to_string( status ) ) );
   if( root )
    f_lb = f_s->get_lb();
   if( f_s->has_true_var_solution() )
    f_incumbent = std::min( f_incumbent , double( f_s->get_true_ub() ) );
   return( f_s->get_lb() < f_incumbent - 1e-9 );
   }

  // from the current node on, depth-first, to the next node to branch on
  // that has a graph; the nodes without one (no unfixed variable, or no
  // clause left) are branched on by the rule of the cores; false when the
  // enumeration is over
  bool advance( bool root ) {
   for( ; ; ) {
    if( evaluate( root ) ) {
     if( f_g.build( *f_s , f_features ) )
      return( true );
     auto children = f_s->branch();
     push( children );
     root = false;
     continue;
     }
    root = false;
    // back up to the first level with a child still to explore
    for( ; ; ) {
     if( f_stack.empty() )
      return( false );
     auto & fr = f_stack.back();
     f_s->apply( fr.undo );
     delete fr.undo;
     if( fr.sibling ) {
      fr.undo = f_s->apply( fr.sibling , true );
      delete fr.sibling;
      fr.sibling = nullptr;
      break;
      }
     f_stack.pop_back();
     }
    }
   }

  // the graph of the current node as numpy arrays: (vertex rows, edge
  // rows, connectivity, global row), empty once the enumeration is over
  py::tuple state( void ) {
   const long nv = f_done ? 0 : f_g.n_var + f_g.n_clause;
   const long ne = f_done ? 0 : long( f_g.source.size() );
   const long nc = SATResidualGraph::n_features( f_features );
   py::array_t< float > v( { nv , nc } );
   py::array_t< float > e( { ne , 2L } );
   py::array_t< int64_t > conn( { 2L , ne } );
   py::array_t< float > u( { 1L , 1L } );
   if( nv )
    std::copy( f_g.vertex.begin() , f_g.vertex.end() , v.mutable_data() );
   if( ne ) {
    std::copy( f_g.edge.begin() , f_g.edge.end() , e.mutable_data() );
    std::copy( f_g.source.begin() , f_g.source.end() , conn.mutable_data() );
    std::copy( f_g.target.begin() , f_g.target.end() ,
               conn.mutable_data() + ne );
    }
   u.mutable_data()[ 0 ] = 0;
   return( py::make_tuple( v , e , conn , u ) );
   }

  void clear( void ) {
   for( auto & fr : f_stack ) {
    delete fr.undo;
    delete fr.sibling;
    }
   f_stack.clear();
   if( f_sat )
    f_sat->unregister_Solvers( true );
   f_s = nullptr;
   f_sat.reset();
   f_g = SATResidualGraph();
   }

  std::string f_solver;
  int f_max_iter;
  int f_features;
  double f_penalty;
  std::unique_ptr< SATBlock > f_sat;
  SATSolver * f_s = nullptr;
  std::vector< Frame > f_stack;
  SATResidualGraph f_g;
  double f_incumbent = Inf< double >();
  double f_lb = - Inf< double >();
  long f_nodes = 0;
  bool f_done = true;
  };

PYBIND11_MODULE( _smspp_env , m )
{
 m.doc() = "the branch and bound of SMS++ on a SATBlock, as an environment";
 py::class_< BnBEnv >( m , "BnBEnv" )
  .def( py::init< const std::string & , int , int , double >() ,
        py::arg( "solver" ) = "CaDiCaLSATSolver" ,
        py::arg( "max_iter" ) = 20 , py::arg( "features" ) = 1 ,
        py::arg( "penalty" ) = 0.1 )
  .def( "reset" , &BnBEnv::reset )
  .def( "step" , &BnBEnv::step )
  .def( "incumbent" , &BnBEnv::incumbent )
  .def( "root_lb" , &BnBEnv::root_lb )
  .def( "nodes" , &BnBEnv::nodes )
  .def( "n_var" , &BnBEnv::n_var )
  .def( "n_col" , &BnBEnv::n_col );
 }
