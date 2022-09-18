#-------------------------------------------------------------------------------
# SuiteSparse/LDL/Makefile
#-------------------------------------------------------------------------------

# LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------

# A simple Makefile for LDL, which relies on cmake to do the
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
	( cd build && cmake $(CMAKE_OPTIONS) .. && $(MAKE) --jobs=${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL=1 .. && $(MAKE) --jobs=${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && $(MAKE) --jobs=${JOBS} )

all: library

demo: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && $(MAKE) --jobs=${JOBS} )
	./build/ldlsimple > ./build/my_ldlsimple.out
	- diff ./Demo/ldlsimple.out ./build/my_ldlsimple.out
	./build/ldllsimple > ./build/my_ldllsimple.out
	- diff ./Demo/ldlsimple.out ./build/my_ldllsimple.out
	./build/ldlmain > ./build/my_ldlmain.out
	- diff ./Demo/ldlmain.out ./build/my_ldlmain.out
	./build/ldllmain > ./build/my_ldllmain.out
	- diff ./Demo/ldlmain.out ./build/my_ldllmain.out
	./build/ldlamd  > ./build/my_ldlamd.out
	- diff ./Demo/ldlamd.out  ./build/my_ldlamd.out
	./build/ldllamd  > ./build/my_ldllamd.out
	- diff ./Demo/ldllamd.out  ./build/my_ldllamd.out

# just compile after running cmake; do not run cmake again
remake:
	( cd build && $(MAKE) --jobs=${JOBS} )

# just run cmake to set things up
setup:
	( cd build && cmake $(CMAKE_OPTIONS) .. )

install:
	( cd build && $(MAKE) install )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# remove all files not in the distribution
clean:
	- $(RM) -rf build/* Config/*.tmp

purge: clean

distclean: clean

docs:
	( cd Doc && $(MAKE) )

