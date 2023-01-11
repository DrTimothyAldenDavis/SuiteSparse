#-------------------------------------------------------------------------------
# SuiteSparse/KLU/Makefile
#-------------------------------------------------------------------------------

# KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
# Authors: Timothy A. Davis and Ekanathan Palamadai.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------

# A simple Makefile for KLU, which relies on cmake to do the
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
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=1 .. && cmake --build . -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=0 .. && cmake --build . -j${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . -j${JOBS} )
	- ./build/klu_simple
	- ./build/kludemo  < ./Matrix/1c.mtx
	- ./build/kludemo  < ./Matrix/arrowc.mtx
	- ./build/kludemo  < ./Matrix/arrow.mtx
	- ./build/kludemo  < ./Matrix/impcol_a.mtx
	- ./build/kludemo  < ./Matrix/w156.mtx
	- ./build/kludemo  < ./Matrix/ctina.mtx
	- ./build/kluldemo < ./Matrix/1c.mtx
	- ./build/kluldemo < ./Matrix/arrowc.mtx
	- ./build/kluldemo < ./Matrix/arrow.mtx
	- ./build/kluldemo < ./Matrix/impcol_a.mtx
	- ./build/kluldemo < ./Matrix/w156.mtx
	- ./build/kluldemo < ./Matrix/ctina.mtx

cov:
	( cd Tcov && $(MAKE) )

# just compile after running cmake; do not run cmake again
remake:
	( cd build && cmake --build . -j${JOBS} )

# just run cmake to set things up
setup:
	( cd build && cmake $(CMAKE_OPTIONS) .. )

install:
	( cd build && cmake --install . )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# remove all files not in the distribution
clean:
	- $(RM) -rf build/* Config/*.tmp MATLAB/*.o MATLAB/*.mex*
	( cd Tcov && $(MAKE) purge )

purge: clean

distclean: clean

docs:
	( cd Doc && $(MAKE) )

