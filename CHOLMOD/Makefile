#-------------------------------------------------------------------------------
# SuiteSparse/CHOLOMD/Makefile
#-------------------------------------------------------------------------------

# This CHOLMOD/Makefile: Copyright (c) 2005-2022, Timothy A. Davis.
# CHOLMOD: Copyright and license varies by module.

#-------------------------------------------------------------------------------

# A simple Makefile for CHOLMOD, which relies on cmake to do the
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

# default is to install only in /usr/local
library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

# enable CUDA (this is the default, if CUDA is available)
cuda:
	( cd build && cmake $(CMAKE_OPTIONS) -DENABLE_CUDA=1 .. && cmake --build . --config Release -j$(JOBS) )

# disable CUDA
nocuda:
	( cd build && cmake $(CMAKE_OPTIONS) -DENABLE_CUDA=0 .. && cmake --build . --config Release -j$(JOBS) )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Debug -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
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
	- $(RM) -rf build/* Config/*.tmp MATLAB/*.o MATLAB/*.mex* timelog.m
	- $(RM) -rf MATLAB/*/*.o MATLAB/*/*.mex*
	( cd Tcov && $(MAKE) purge )

purge: clean

distclean: clean

docs:
	( cd Doc && $(MAKE) )

