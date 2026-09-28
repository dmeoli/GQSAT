# the Python module of the branch and bound of SMS++ on a SATBlock, built
# against an installed SATBlock (CMAKE_PREFIX_PATH) and pybind11
cmake_minimum_required(VERSION 3.19)
project(smspp_env CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(SATBlock REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_smspp_env bnb_env.cpp)
target_link_libraries(_smspp_env PRIVATE SMS++::SATBlock)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(_smspp_env PRIVATE "LINKER:--no-as-needed")
endif ()
