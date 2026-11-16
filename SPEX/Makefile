#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/Makefile: Makefile for SPEX
#-------------------------------------------------------------------------------

# SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#-------------------------------------------------------------------------------

# A simple Makefile for SPEX, which relies on cmake to do the
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

demos:
	( cd build && cmake $(CMAKE_OPTIONS) -DSUITESPARSE_DEMOS=1 .. && cmake --build . --config Release -j${JOBS} )
	./build/spex_demo_lu_simple1
	./build/spex_demo_lu_simple2          ExampleMats/10teams.mat.txt   ExampleMats/10teams.rhs.txt
	./build/spex_demo_lu_extended       f ExampleMats/10teams.mat.txt   ExampleMats/10teams.rhs.txt
	./build/spex_demo_lu_doub           f ExampleMats/10teams.mat.txt   ExampleMats/10teams.rhs.txt
	./build/spex_demo_backslash         f ExampleMats/10teams.mat.txt   ExampleMats/10teams.rhs.txt
	./build/spex_demo_cholesky_simple   f ExampleMats/494_bus.mat.txt   ExampleMats/494_bus.rhs.txt
	./build/spex_demo_cholesky_extended f ExampleMats/494_bus.mat.txt   ExampleMats/494_bus.rhs.txt
	./build/spex_demo_ldl_simple        f ExampleMats/494_bus.mat.txt   ExampleMats/494_bus.rhs.txt
	./build/spex_demo_ldl_extended      f ExampleMats/494_bus.mat.txt   ExampleMats/494_bus.rhs.txt
	./build/spex_demo_threaded          f ExampleMats/10teams.mat.txt   ExampleMats/10teams.rhs.txt
	./build/spex_demo_backslash         f ExampleMats/Trefethen_500.mat.txt ExampleMats/Trefethen_500.rhs.txt
	./build/spex_demo_qr_simple 		f ExampleMats/LF10.mat.txt ExampleMats/LF10.rhs.txt
	./build/spex_demo_qr_extended 		f ExampleMats/smallZeros.mat.txt ExampleMats/smallZeros.rhs.txt
	./build/spex_demo_qr_cluster 		f ExampleMats/smallZeros.mat.txt ExampleMats/smallZeros.rhs.txt
	./build/spex_demo_qr

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
	- $(RM) Tcov/*.o Tcov/*.bbg Tcov/*.da Tcov/*.gcov Tcov/*.gcda Tcov/*gcno
	- $(RM) Tcov/tcov_for_lu  Tcov/spex_demo_lu_extended Tcov/*.out Tcov/*.a Tcov/out Tcov/tcov_for_lu2 Tcov/tcov_for_other
	- $(RM) Tcov/tcov_for_update Tcov/out Tcov/Mats4Tcov/SPEX_Update/mat.txt Tcov/tcov_for_qr
	- $(RM) -r Tcov/SPEX_*.c Tcov/spex_*.c Tcov/*.dSYM
	- $(RM) -r Tcov/tcov_for_cholesky

purge: clean

distclean: clean

docs:
	( cd Doc && $(MAKE) )
