#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# A simple Makefile for SuiteSparse_config, which relies on cmake to do the
# actual build.  All the work is done in cmake so this Makefile is just for
# convenience.

# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To compile/install for system-wide usage and for local usage in
# SuiteSparse/lib and SuiteSparse/include:
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

debug:
	( cd build ; cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. ; $(MAKE) )

all: library

demos: library

# just compile after running cmake; do not run cmake again
remake:
	( cd build ; $(MAKE) )

# just run cmake to set things up
setup:
	( cd build ; cmake $(CMAKE_OPTIONS) .. )

install:
	( cd build ; $(MAKE) install )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# remove all files not in the distribution
clean: distclean

purge: distclean

distclean:
	- $(RM) -rf build/* Config/*.tmp

docs:

