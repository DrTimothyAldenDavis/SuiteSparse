#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_metis/Makefile
#-------------------------------------------------------------------------------

# METIS, Copyright 1995-2013, Regents of the University of Minnesota.
# Author: George Karypis
# SPDX-License-identifier: Apache-2.0

# This Makefile was added by Tim Davis, Oct 2022,
# to build SuiteSparse_metis for use in SuiteSparse.  See
# include/SuiteSparse_metis.h for a description of the changes made in this
# version, as compared to the original METIS v5.1.0.

# The original METIS v5.1.0 Makefile has been moved to original/Makefile.

#-------------------------------------------------------------------------------

# A simple Makefile for SuiteSparse_metis, which relies on cmake to do the
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
	( cd build && cmake $(CMAKE_OPTIONS) -DSUITESPARSE_LOCAL=1 .. && $(MAKE) --jobs=${JOBS} )

# install in SuiteSparse/lib and SuiteSparse/include and /usr/local
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DSUITESPARSE_LOCAL=0 .. && $(MAKE) --jobs=${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && $(MAKE) --jobs=${JOBS} )

all: library

# metis comes with itws own programs, but this Makefile doesn't build them 
demo: library

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
	- $(RM) -rf build/*

purge: clean

distclean: clean

docs:

