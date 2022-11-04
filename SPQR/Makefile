#-------------------------------------------------------------------------------
# SuiteSparseQR Makefile
#-------------------------------------------------------------------------------

# SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------

# A simple Makefile for SuiteSparseQR, which relies on cmake to do the
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
	( cd build && cmake $(CMAKE_OPTIONS) .. && $(MAKE) --jobs=${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DGLOBAL_INSTALL=0 -DLOCAL_INSTALL=1 .. && $(MAKE) --jobs=${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DGLOBAL_INSTALL=1 -DLOCAL_INSTALL=0 .. && $(MAKE) --jobs=${JOBS} )

# install in SuiteSparse/lib both and SuiteSparse/include and /usr/local
both:
	( cd build && cmake $(CMAKE_OPTIONS) -DGLOBAL_INSTALL=1 -DLOCAL_INSTALL=1 .. && $(MAKE) --jobs=${JOBS} )

# enable CUDA (this is the default, if CUDA is available)
cuda:
	( cd build && cmake $(CMAKE_OPTIONS) -DENABLE_CUDA=1 .. && $(MAKE) --jobs=$(JOBS) )

# disable CUDA
nocuda:
	( cd build && cmake $(CMAKE_OPTIONS) -DENABLE_CUDA=0 .. && $(MAKE) --jobs=$(JOBS) )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && $(MAKE) --jobs=${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && $(MAKE) --jobs=${JOBS} )
	- ./build/qrsimple  < Matrix/ash219.mtx
	- ./build/qrsimplec < Matrix/ash219.mtx
	- ./build/qrsimple  < Matrix/west0067.mtx
	- ./build/qrsimplec < Matrix/west0067.mtx
	- ./build/qrdemo < Matrix/a2.mtx
	- ./build/qrdemo < Matrix/r2.mtx
	- ./build/qrdemo < Matrix/a04.mtx
	- ./build/qrdemo < Matrix/a2.mtx
	- ./build/qrdemo < Matrix/west0067.mtx
	- ./build/qrdemo < Matrix/c2.mtx
	- ./build/qrdemo < Matrix/a0.mtx
	- ./build/qrdemo < Matrix/lfat5b.mtx
	- ./build/qrdemo < Matrix/bfwa62.mtx
	- ./build/qrdemo < Matrix/LFAT5.mtx
	- ./build/qrdemo < Matrix/b1_ss.mtx
	- ./build/qrdemo < Matrix/bcspwr01.mtx
	- ./build/qrdemo < Matrix/lpi_galenet.mtx
	- ./build/qrdemo < Matrix/lpi_itest6.mtx
	- ./build/qrdemo < Matrix/ash219.mtx
	- ./build/qrdemo < Matrix/a4.mtx
	- ./build/qrdemo < Matrix/s32.mtx
	- ./build/qrdemo < Matrix/c32.mtx
	- ./build/qrdemo < Matrix/lp_share1b.mtx
	- ./build/qrdemo < Matrix/a1.mtx
	- ./build/qrdemo < Matrix/GD06_theory.mtx
	- ./build/qrdemo < Matrix/GD01_b.mtx
	- ./build/qrdemo < Matrix/Tina_AskCal_perm.mtx
	- ./build/qrdemo < Matrix/Tina_AskCal.mtx
	- ./build/qrdemo < Matrix/GD98_a.mtx
	- ./build/qrdemo < Matrix/Ragusa16.mtx
	- ./build/qrdemo < Matrix/young1c.mtx
	- ./build/qrdemo < Matrix/lp_e226_transposed.mtx
	- ./build/qrdemoc < Matrix/a2.mtx
	- ./build/qrdemoc < Matrix/r2.mtx
	- ./build/qrdemoc < Matrix/a04.mtx
	- ./build/qrdemoc < Matrix/a2.mtx
	- ./build/qrdemoc < Matrix/west0067.mtx
	- ./build/qrdemoc < Matrix/c2.mtx
	- ./build/qrdemoc < Matrix/a0.mtx
	- ./build/qrdemoc < Matrix/lfat5b.mtx
	- ./build/qrdemoc < Matrix/bfwa62.mtx
	- ./build/qrdemoc < Matrix/LFAT5.mtx
	- ./build/qrdemoc < Matrix/b1_ss.mtx
	- ./build/qrdemoc < Matrix/bcspwr01.mtx
	- ./build/qrdemoc < Matrix/lpi_galenet.mtx
	- ./build/qrdemoc < Matrix/lpi_itest6.mtx
	- ./build/qrdemoc < Matrix/ash219.mtx
	- ./build/qrdemoc < Matrix/a4.mtx
	- ./build/qrdemoc < Matrix/s32.mtx
	- ./build/qrdemoc < Matrix/c32.mtx
	- ./build/qrdemoc < Matrix/lp_share1b.mtx
	- ./build/qrdemoc < Matrix/a1.mtx
	- ./build/qrdemoc < Matrix/GD06_theory.mtx
	- ./build/qrdemoc < Matrix/GD01_b.mtx
	- ./build/qrdemoc < Matrix/Tina_AskCal_perm.mtx
	- ./build/qrdemoc < Matrix/Tina_AskCal.mtx
	- ./build/qrdemoc < Matrix/GD98_a.mtx
	- ./build/qrdemoc < Matrix/Ragusa16.mtx
	- ./build/qrdemoc < Matrix/young1c.mtx
	- ./build/qrdemoc < Matrix/lp_e226_transposed.mtx

cov:
	( cd Tcov && $(MAKE) )

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
	- $(RM) -rf build/* Config/*.tmp MATLAB/*.o MATLAB/*.mex* timelog.m
	- $(RM) -rf MATLAB/*/*.o MATLAB/*/*.mex*
	- $(RM) -rf Demo/o_* Demo/A.mtx  Demo/X.mtx Demo/gpu_results.txt
	( cd Tcov && $(MAKE) purge )

purge: clean

distclean: clean

docs:
	( cd Doc && $(MAKE) )

