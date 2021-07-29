/*
This is the main interface for python to communicate with the solver. In the original implementation
the state was turned into a string which was later parsed on the python side.
We have rewritten this and return python lists instead. One step further would be to return numpy arrays, which
should potentially be faster
*/

%module GymSolver

%include <stdint.i>
// Graph-Q-SAT UPD: return vectors to support variable sized lists. The original version had a fixed sized arrays.
%typemap(out) std::vector<int> *get_metadata %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
    PyList_SetItem($result, i, PyLong_FromLong((*$1)[i]));
   }
%}

%typemap(out) std::vector<int> *get_assignments %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
    PyList_SetItem($result, i, PyLong_FromLong((*$1)[i]));
   }
%}

%typemap(out) std::vector<double> *get_activities %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
    PyList_SetItem($result, i, PyFloat_FromDouble((*$1)[i]));
   }
%}

%typemap(out) std::vector<std::vector <int> > *get_clauses %{
    $result = PyList_New($1->size());
    for (int i = 0; i < $1->size(); ++i) {
        std::vector <int> curr_clause_vec = (*$1)[i];
        int clause_size = curr_clause_vec.size();

        PyObject* curr_clause = PyList_New(clause_size);
        for (int j = 0; j < clause_size; ++j) {
            PyList_SetItem(curr_clause, j, PyLong_FromLong(curr_clause_vec[j]));
        }
        PyList_SetItem($result, i, curr_clause);
    }
%}

// to solve a problem given an adjacency matrix
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* adj_mat, int cla_cnt, int var_cnt)};

// end of Graph-Q-SAT UPD.
%include "GymSolver.h"

%{
#include <zlib.h>
#include "GymSolver.h"
%}
