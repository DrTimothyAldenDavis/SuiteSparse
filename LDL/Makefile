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

# default is to install only in /usr/local
library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DSUITESPARSE_LOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DSUITESPARSE_LOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Debug -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DSUITESPARSE_DEMOS=1 .. && cmake --build . --config Release -j${JOBS} )
	./build/ldlsimple > ./build/ldlsimple.out && ( command -v d2u && d2u ./build/ldlsimple.out || true )
	- diff ./Demo/ldlsimple.out ./build/ldlsimple.out
	./build/ldllsimple > ./build/ldllsimple.out && ( command -v d2u && d2u ./build/ldllsimple.out || true )
	- diff ./Demo/ldllsimple.out ./build/ldllsimple.out
	./build/ldlmain > ./build/ldlmain.out && ( command -v d2u && d2u ./build/ldlmain.out || true )
	- diff ./Demo/ldlmain.out ./build/ldlmain.out
	./build/ldllmain > ./build/ldllmain.out && ( command -v d2u && d2u ./build/ldllmain.out || true )
	- diff ./Demo/ldllmain.out ./build/ldllmain.out
	./build/ldlamd  > ./build/ldlamd.out && ( command -v d2u && d2u ./build/ldlamd.out || true )
	- diff ./Demo/ldlamd.out  ./build/ldlamd.out
	./build/ldllamd  > ./build/ldllamd.out && ( command -v d2u && d2u ./build/ldllamd.out || true )
	- diff ./Demo/ldllamd.out  ./build/ldllamd.out

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

purge: clean

distclean: clean

docs:
	( cd Doc && $(MAKE) )

