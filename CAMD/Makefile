#-------------------------------------------------------------------------------
# SuiteSparse/CAMD/Makefile
#-------------------------------------------------------------------------------

# CAMD: Copyright (c) 2009-2022, Timothy A. Davis, Patrick Amestoy, Iain Duff,
# Yanqing Chen.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# A simple Makefile for CAMD, which relies on cmake to do the
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
	./build/camd_demo > build/camd_demo.out && ( command -v d2u && d2u ./build/amd_demo.out || true )
	- diff Demo/camd_demo.out build/camd_demo.camd_demo
	./build/camd_l_demo > build/camd_l_demo.out && ( command -v d2u && d2u ./build/camd_l_demo.out || true )
	- diff Demo/camd_l_demo.out build/camd_l_demo.out
	./build/camd_demo2 > build/camd_demo2.out && ( command -v d2u && d2u ./build/camd_demo2.out || true )
	- diff Demo/camd_demo2.out build/camd_demo2.out
	./build/camd_simple > build/camd_simple.out && ( command -v d2u && d2u ./build/camd_simple.out || true )
	- diff Demo/camd_simple.out build/camd_simple.out

# just compile after running cmake&& do not run cmake again
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

