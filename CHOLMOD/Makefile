#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/Makefile
#-------------------------------------------------------------------------------

# This CHOLMOD/Makefile: Copyright (c) 2005-2023, Timothy A. Davis.
# CHOLMOD: Copyright and license varies by module.

#-------------------------------------------------------------------------------

# A simple Makefile for CHOLMOD, which relies on cmake to do the
# actual build.  All the work is done in cmake so this Makefile is just for
# convenience.

# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To compile with no GPL-licensed modules (MatrixOps, Modify, and Supernodal):
#
#       make CMAKE_OPTIONS='-DCHOLMOD_GPL=OFF'
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

# enable CUDA (this is the default, if CUDA is available)
cuda:
	( cd build && cmake $(CMAKE_OPTIONS) -DCHOLMOD_USE_CUDA=1 .. && cmake --build . --config Release -j$(JOBS) )

# disable CUDA
nocuda:
	( cd build && cmake $(CMAKE_OPTIONS) -DCHOLMOD_USE_CUDA=0 .. && cmake --build . --config Release -j$(JOBS) )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Debug -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DSUITESPARSE_DEMOS=1 .. && cmake --build . --config Release -j${JOBS} )
	./build/cholmod_di_demo ./Demo/Matrix/bcsstk01.tri  > ./build/bcsstk01_out.txt
	./build/cholmod_dl_demo ./Demo/Matrix/bcsstk01.tri >> ./build/bcsstk01_out.txt
	./build/cholmod_si_demo ./Demo/Matrix/bcsstk01.tri >> ./build/bcsstk01_out.txt
	./build/cholmod_sl_demo ./Demo/Matrix/bcsstk01.tri >> ./build/bcsstk01_out.txt
	#
	./build/cholmod_di_demo ./Demo/Matrix/lp_afiro.tri  > ./build/lp_afiro_out.txt
	./build/cholmod_dl_demo ./Demo/Matrix/lp_afiro.tri >> ./build/lp_afiro_out.txt
	./build/cholmod_si_demo ./Demo/Matrix/lp_afiro.tri >> ./build/lp_afiro_out.txt
	./build/cholmod_sl_demo ./Demo/Matrix/lp_afiro.tri >> ./build/lp_afiro_out.txt
	#
	./build/cholmod_di_demo ./Demo/Matrix/can___24.mtx  > ./build/can24_out.txt
	./build/cholmod_dl_demo ./Demo/Matrix/can___24.mtx >> ./build/can24_out.txt
	./build/cholmod_si_demo ./Demo/Matrix/can___24.mtx >> ./build/can24_out.txt
	./build/cholmod_sl_demo ./Demo/Matrix/can___24.mtx >> ./build/can24_out.txt
	#
	./build/cholmod_di_demo ./Demo/Matrix/c.tri         > ./build/complex_out.txt
	./build/cholmod_dl_demo ./Demo/Matrix/c.tri        >> ./build/complex_out.txt
	./build/cholmod_si_demo ./Demo/Matrix/c.tri        >> ./build/complex_out.txt
	./build/cholmod_sl_demo ./Demo/Matrix/c.tri        >> ./build/complex_out.txt
	#
	./build/cholmod_di_simple < ./Demo/Matrix/bcsstk01.tri > ./build/di_simple.out.txt
	./build/cholmod_dl_simple < ./Demo/Matrix/bcsstk01.tri > ./build/dl_simple.out.txt
	./build/cholmod_si_simple < ./Demo/Matrix/bcsstk01.tri > ./build/si_simple.out.txt
	./build/cholmod_sl_simple < ./Demo/Matrix/bcsstk01.tri > ./build/sl_simple.out.txt
	#
	./build/cholmod_di_simple < ./Demo/Matrix/lp_afiro.tri >> ./build/di_simple.out.txt
	./build/cholmod_dl_simple < ./Demo/Matrix/lp_afiro.tri >> ./build/dl_simple.out.txt
	./build/cholmod_si_simple < ./Demo/Matrix/lp_afiro.tri >> ./build/si_simple.out.txt
	./build/cholmod_sl_simple < ./Demo/Matrix/lp_afiro.tri >> ./build/sl_simple.out.txt
	#
	./build/cholmod_di_simple < ./Demo/Matrix/can___24.mtx >> ./build/di_simple.out.txt
	./build/cholmod_dl_simple < ./Demo/Matrix/can___24.mtx >> ./build/dl_simple.out.txt
	./build/cholmod_si_simple < ./Demo/Matrix/can___24.mtx >> ./build/si_simple.out.txt
	./build/cholmod_sl_simple < ./Demo/Matrix/can___24.mtx >> ./build/sl_simple.out.txt
	#
	./build/cholmod_di_simple < ./Demo/Matrix/c.tri        >> ./build/di_simple.out.txt
	./build/cholmod_dl_simple < ./Demo/Matrix/c.tri        >> ./build/dl_simple.out.txt
	./build/cholmod_si_simple < ./Demo/Matrix/c.tri        >> ./build/si_simple.out.txt
	./build/cholmod_sl_simple < ./Demo/Matrix/c.tri        >> ./build/sl_simple.out.txt

large: library
	./build/cholmod_dl_simple < ~/nd12k.mtx > ./build/dl_nd12k.out.txt
	./build/cholmod_sl_simple < ~/nd12k.mtx > ./build/sl_nd12k.out.txt

ilarge: library
	./build/cholmod_di_simple < ~/nd12k.mtx > ./build/di_nd12k.out.txt
	./build/cholmod_si_simple < ~/nd12k.mtx > ./build/si_nd12k.out.txt

test: library
	( cd build && ctest . || ctest . --rerun-failed --output-on-failure )

cov:
	( cd Tcov && $(MAKE) )

gpu: library
	( cd Demo && ./gpu.sh )

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

