#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/Makefile
#-------------------------------------------------------------------------------

# UMFPACK: Copyright (c) 1995-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------

# A simple Makefile for UMFPACK, which relies on cmake to do the
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
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Debug -j${JOBS} )

all: library

demos: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
	( cd build && ./umfpack_simple )
	- ( cd build && ./umfpack_di_demo > umfpack_di_demo.out && diff --strip-trailing-cr ../Demo/umfpack_di_demo.out umfpack_di_demo.out )
	- ( cd build && ./umfpack_dl_demo > umfpack_dl_demo.out && diff --strip-trailing-cr ../Demo/umfpack_dl_demo.out umfpack_dl_demo.out )
	- ( cd build && ./umfpack_zi_demo > umfpack_zi_demo.out && diff --strip-trailing-cr ../Demo/umfpack_zi_demo.out umfpack_zi_demo.out )
	- ( cd build && ./umfpack_zl_demo > umfpack_zl_demo.out && diff --strip-trailing-cr ../Demo/umfpack_zl_demo.out umfpack_zl_demo.out )

# demos requiring a Fortran compiler
fdemos: demos
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && cmake --build . --config Release -j${JOBS} )
	( cd build && ./readhb_nozeros < ../Demo/HB/can_24.psa > tmp_A          )
	( cd build && ./readhb_size    < ../Demo/HB/can_24.psa > tmp_Asize      )
	( cd build && ./umf4                                                    )
	( cd build && ./readhb_nozeros < ../Demo/HB/west0067.rua > tmp_A        )
	( cd build && ./readhb_size    < ../Demo/HB/west0067.rua > tmp_Asize    )
	( cd build && ./umf4                                                    )
	( cd build && ./readhb_nozeros < ../Demo/HB/fs_183_6.rua > tmp_A        )
	( cd build && ./readhb_size    < ../Demo/HB/fs_183_6.rua > tmp_Asize    )
	( cd build && ./umf4                                                    )
	( cd build && ./readhb         < ../Demo/HB/fs_183_6.rua > tmp_A        )
	( cd build && ./readhb_size    < ../Demo/HB/fs_183_6.rua > tmp_Asize    )
	( cd build && ./umf4                                                    )
	( cd build && ./readhb         < ../Demo/HB/arc130.rua > tmp_A          )
	( cd build && ./readhb_size    < ../Demo/HB/arc130.rua > tmp_Asize      )
	( cd build && ./umf4                                                    )
	( cd build && ./readhb_nozeros < ../Demo/HB/arc130.rua > tmp_A          )
	( cd build && ./readhb_size    < ../Demo/HB/arc130.rua > tmp_Asize      )
	( cd build && ./umf4                                                    )
	( cd build && ./readhb_nozeros < ../Demo/HB/arc130.rua > tmp_A          )
	( cd build && ./readhb_size    < ../Demo/HB/arc130.rua > tmp_Asize      )
	( cd build && ./umf4 a 1e-6                                             )
	( cd build && ./umf4hb         < ../Demo/HB/west0067.rua > umf4hb.out   )
	- diff --strip-trailing-cr ./Demo/umf4hb.out  ./build/umf4hb.out 
	( cd build && ./umf4hb64       < ../Demo/HB/west0067.rua > umf4hb64.out )
	- diff --strip-trailing-cr ./Demo/umf4hb64.out  ./build/umf4hb64.out 
	( cd build && ./umf4zhb        < ../Demo/HB/qc324.cua > umf4zhb.out     )
	- diff --strip-trailing-cr ./Demo/umf4zhb.out ./build/umf4zhb.out
	( cd build && ./umf4zhb64      < ../Demo/HB/qc324.cua > umf4zhb64.out   )
	- diff --strip-trailing-cr ./Demo/umf4zhb64.out ./build/umf4zhb64.out

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

