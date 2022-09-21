#-------------------------------------------------------------------------------
# SuiteSparse/COLAMD/Makefile
#-------------------------------------------------------------------------------

# Copyright (c) 1998-2022, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# A simple Makefile for COLAMD, which relies on cmake to do the
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

library:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; $(MAKE) --jobs=${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build ; cmake $(CMAKE_OPTIONS) -DSUITESPARSE_LOCAL=1 .. ; $(MAKE) --jobs=${JOBS} )

debug:
	( cd build ; cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. ; $(MAKE) --jobs=${JOBS} )

all: library

demo: library
	( cd build ; cmake $(CMAKE_OPTIONS) -DDEMO=1 .. ; $(MAKE) --jobs=${JOBS} )
	- ./build/colamd_example > ./build/my_colamd_example.out
	- diff ./Demo/colamd_example.out ./build/my_colamd_example.out
	- ./build/colamd_l_example > ./build/my_colamd_l_example.out
	- diff ./Demo/colamd_l_example.out ./build/my_colamd_l_example.out

# just compile after running cmake; do not run cmake again
remake:
	( cd build ; $(MAKE) --jobs=${JOBS} )

# just run cmake to set things up
setup:
	( cd build ; cmake $(CMAKE_OPTIONS) .. )

install:
	( cd build ; $(MAKE) install )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# remove all files not in the distribution
clean:
	- $(RM) -rf build/* Config/*.tmp MATLAB/*.o MATLAB/*.mex* timelog.m

purge: clean

distclean: clean

