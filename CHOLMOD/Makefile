#-------------------------------------------------------------------------------
# SuiteSparse/CHOLOMD/Makefile
#-------------------------------------------------------------------------------

# CHOLMOD: Copyright (c) 2005-2022, Timothy A. Davis.
# Copyright and license varies by module.

#-------------------------------------------------------------------------------

# A simple Makefile for CHOLOMD, which relies on cmake to do the
# actual build.  All the work is done in cmake so this Makefile is just for
# convenience.

# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To compile with any GPL modules (MatrixOps, Modify, and Supernodal):
#
#       make CMAKE_OPTIONS='-DNGPL=1'
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
	echo "CMake options: " $(CMAKE_OPTIONS)
	( cd build && cmake $(CMAKE_OPTIONS) .. && $(MAKE) --jobs=${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL=1 .. && $(MAKE) --jobs=${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && $(MAKE) --jobs=${JOBS} )

all: library

demo: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && $(MAKE) --jobs=${JOBS} )
	./build/cholmod_demo   < ./Demo/Matrix/bcsstk01.tri
	./build/cholmod_l_demo < ./Demo/Matrix/bcsstk01.tri
	./build/cholmod_demo   < ./Demo/Matrix/lp_afiro.tri
	./build/cholmod_l_demo < ./Demo/Matrix/lp_afiro.tri
	./build/cholmod_demo   < ./Demo/Matrix/can___24.mtx
	./build/cholmod_l_demo < ./Demo/Matrix/can___24.mtx
	./build/cholmod_demo   < ./Demo/Matrix/c.tri
	./build/cholmod_l_demo < ./Demo/Matrix/c.tri
	./build/cholmod_simple < ./Demo/Matrix/c.tri
	./build/cholmod_simple < ./Demo/Matrix/can___24.mtx
	./build/cholmod_simple < ./Demo/Matrix/bcsstk01.tri

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
	- $(RM) -rf build/* Config/*.tmp timelog.m

purge: clean

distclean: clean

