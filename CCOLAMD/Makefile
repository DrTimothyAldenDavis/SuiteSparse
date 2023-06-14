#-------------------------------------------------------------------------------
# SuiteSparse/CCOLAMD/Makefile
#-------------------------------------------------------------------------------

# CCOLAMD: Copyright (c) 2005-2016.  Authors: Timothy A. Davis,
# Sivasankaran Rajamanickam, and Stefan Larimore.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# A simple Makefile for CCOLAMD, which relies on cmake to do the
# actual build.  All the work is done in cmake so this Makefile is just for
# convenience.

# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To compile/install for system-wide usage:
#
#       make
#       sudo make install
#
# To compile/install for local usage (SuiteSparse/lib and SuiteSparse/include):
#
#       make local
#       make install
#
# To clean up the files:
#
#       make clean

JOBS ?= 8

default: library

# default is to install only in /usr/local
library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Debug -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
	- ./build/ccolamd_example > ./build/ccolamd_example.out
	- diff --strip-trailing-cr ./Demo/ccolamd_example.out ./build/ccolamd_example.out
	- ./build/ccolamd_l_example > ./build/ccolamd_l_example.out
	- diff --strip-trailing-cr ./Demo/ccolamd_l_example.out ./build/ccolamd_l_example.out

# just compile after running cmake; do not run cmake again
remake:
	( cd build && cmake --build . -j${JOBS} )

# just run cmake to set things up
setup:
	( cd build ; cmake $(CMAKE_OPTIONS) .. )

install:
	( cd build && cmake --install . )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# remove all files not in the distribution
clean:
	- $(RM) -rf build/* Config/*.tmp MATLAB/*.o MATLAB/*.mex*

purge: clean

distclean: clean

docs:

