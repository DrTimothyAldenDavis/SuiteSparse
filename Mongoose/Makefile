#-------------------------------------------------------------------------------
# Mongoose/Makefile
#-------------------------------------------------------------------------------

# Mongoose, Timothy A. Davis, Scott P. Kolodziej, William W. Hager,
# S. Nuri Yeralan, (c) 2017-2018, All Rights Reserved.
# http://suitesparse.com   See Mongoose/Doc/License.txt for license.

#-------------------------------------------------------------------------------

# simple Makefile for Mongoose, relies on cmake to do the actual build.  Use
# the CMAKE_OPTIONS argument to this Makefile to pass options to cmake.

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

# build the Mongoose library (static and dynamic) and run a quick test
demo: library
	( cd build ; ./bin/demo )

# just build the static and dynamic libraries; do not run the demo
library:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; $(MAKE) --jobs=$(JOBS) )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build ; cmake $(CMAKE_OPTIONS) -DSUITESPARSE_LOCAL=1 .. ; $(MAKE) --jobs=${JOBS} )

# install in SuiteSparse/lib and SuiteSparse/include and /usr/local
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DSUITESPARSE_LOCAL=0 .. && $(MAKE) --jobs=${JOBS} )

# the same as "make library"
static: library

# installs Mongoose to the install location defined by cmake, usually
# /usr/local/lib and /usr/local/include
install:
	( cd build && $(MAKE) install )

# create the Doc/Mongoose_UserGuide.pdf
docs:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; $(MAKE) userguide )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# run the extensive tests
# FIXME: Mongoose 'make test' fails: unable to download test files
# because of a change in python.
test:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; $(MAKE) test )

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	- $(RM) -rf build/* Version/*.tmp MATLAB/*.o MATLAB/*.mex*

