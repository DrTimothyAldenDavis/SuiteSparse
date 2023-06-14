#-------------------------------------------------------------------------------
# CSparse/Makefile
#-------------------------------------------------------------------------------

# CSparse: Copyright (c) 2006-2022, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------

# A simple Makefile for CSparse, which relies on cmake to do the
# actual build.  All the work is done in cmake so this Makefile is just for
# convenience.

# Note that there is no "make install" target.  For production use, CXSparse
# should be installed instead.

# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To compile:
#
#       make
#
# To clean up the files:
#
#       make clean
#
# To run the demos
#
#       make demos
#
# To run test coverage:
#
#       make cov

JOBS ?= 8

default: library

library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

local: library

global: library

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Debug -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
	- ./build/cs_demo1 < ./Matrix/t1
	- ./build/cs_demo2 < ./Matrix/t1
	- ./build/cs_demo2 < ./Matrix/ash219
	- ./build/cs_demo2 < ./Matrix/bcsstk01
	- ./build/cs_demo2 < ./Matrix/fs_183_1
	- ./build/cs_demo2 < ./Matrix/mbeacxc
	- ./build/cs_demo2 < ./Matrix/west0067
	- ./build/cs_demo2 < ./Matrix/lp_afiro
	- ./build/cs_demo2 < ./Matrix/bcsstk16
	- ./build/cs_demo3 < ./Matrix/bcsstk01
	- ./build/cs_demo3 < ./Matrix/bcsstk16

install:
	- echo 'CSparse is not installed; use CXSparse instead'

# just compile after running cmake; do not run cmake again
remake:
	( cd build && cmake --build . -j${JOBS} )

# just run cmake to set things up
setup:
	( cd build && cmake $(CMAKE_OPTIONS) .. )

# remove all files not in the distribution
clean:
	- $(RM) -rf build/* Config/*.tmp MATLAB/*.o MATLAB/*.mex* timelog.m
	- $(RM) -rf MATLAB/*/*.o MATLAB/*/*.mex*
	( cd Tcov && $(MAKE) purge )

purge: clean

distclean: clean

# test coverage
cov:
	( cd Tcov && $(MAKE) )

