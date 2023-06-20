#-------------------------------------------------------------------------------
# CXSparse/Makefile
#-------------------------------------------------------------------------------

# CXSparse: Copyright (c) 2009-2022, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------

# A simple Makefile for CXSparse, which relies on cmake to do the
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
#
# To run test coverage:
#
#       make cov

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

demos:  rdemos cdemos

# demos for real data type
rdemos:
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
	- ./build/cs_demo1 < Matrix/t1
	- ./build/cs_demo2 < Matrix/t1
	- ./build/cs_demo2 < Matrix/fs_183_1
	- ./build/cs_demo2 < Matrix/west0067
	- ./build/cs_demo2 < Matrix/lp_afiro
	- ./build/cs_demo2 < Matrix/ash219
	- ./build/cs_demo2 < Matrix/mbeacxc
	- ./build/cs_demo2 < Matrix/bcsstk01
	- ./build/cs_demo3 < Matrix/bcsstk01
	- ./build/cs_demo2 < Matrix/bcsstk16
	- ./build/cs_demo3 < Matrix/bcsstk16
	- ./build/cs_di_demo1 < Matrix/t1
	- ./build/cs_di_demo2 < Matrix/t1
	- ./build/cs_di_demo2 < Matrix/fs_183_1
	- ./build/cs_di_demo2 < Matrix/west0067
	- ./build/cs_di_demo2 < Matrix/lp_afiro
	- ./build/cs_di_demo2 < Matrix/ash219
	- ./build/cs_di_demo2 < Matrix/mbeacxc
	- ./build/cs_di_demo2 < Matrix/bcsstk01
	- ./build/cs_di_demo3 < Matrix/bcsstk01
	- ./build/cs_di_demo2 < Matrix/bcsstk16
	- ./build/cs_di_demo3 < Matrix/bcsstk16
	- ./build/cs_dl_demo1 < Matrix/t1
	- ./build/cs_dl_demo2 < Matrix/t1
	- ./build/cs_dl_demo2 < Matrix/fs_183_1
	- ./build/cs_dl_demo2 < Matrix/west0067
	- ./build/cs_dl_demo2 < Matrix/lp_afiro
	- ./build/cs_dl_demo2 < Matrix/ash219
	- ./build/cs_dl_demo2 < Matrix/mbeacxc
	- ./build/cs_dl_demo2 < Matrix/bcsstk01
	- ./build/cs_dl_demo3 < Matrix/bcsstk01
	- ./build/cs_dl_demo2 < Matrix/bcsstk16
	- ./build/cs_dl_demo3 < Matrix/bcsstk16

# demos for complex data type
cdemos:
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
	- ./build/cs_idemo < Matrix/t2
	- ./build/cs_ldemo < Matrix/t2
	- ./build/cs_ci_demo1 < Matrix/t2
	- ./build/cs_ci_demo2 < Matrix/t2
	- ./build/cs_ci_demo2 < Matrix/t3
	- ./build/cs_ci_demo2 < Matrix/t4
	- ./build/cs_ci_demo2 < Matrix/c_west0067
	- ./build/cs_ci_demo2 < Matrix/c_mbeacxc
	- ./build/cs_ci_demo2 < Matrix/young1c
	- ./build/cs_ci_demo2 < Matrix/qc324
	- ./build/cs_ci_demo2 < Matrix/neumann
	- ./build/cs_ci_demo2 < Matrix/c4
	- ./build/cs_ci_demo3 < Matrix/c4
	- ./build/cs_ci_demo2 < Matrix/mhd1280b
	- ./build/cs_ci_demo3 < Matrix/mhd1280b
	- ./build/cs_cl_demo1 < Matrix/t2
	- ./build/cs_cl_demo2 < Matrix/t2
	- ./build/cs_cl_demo2 < Matrix/t3
	- ./build/cs_cl_demo2 < Matrix/t4
	- ./build/cs_cl_demo2 < Matrix/c_west0067
	- ./build/cs_cl_demo2 < Matrix/c_mbeacxc
	- ./build/cs_cl_demo2 < Matrix/young1c
	- ./build/cs_cl_demo2 < Matrix/qc324
	- ./build/cs_cl_demo2 < Matrix/neumann
	- ./build/cs_cl_demo2 < Matrix/c4
	- ./build/cs_cl_demo3 < Matrix/c4
	- ./build/cs_cl_demo2 < Matrix/mhd1280b
	- ./build/cs_cl_demo3 < Matrix/mhd1280b

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

# test coverage
cov:
	( cd Tcov && $(MAKE) )

