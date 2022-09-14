#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/Makefile
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2022, Timothy A. Davis.  All Rights Reserved.
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

library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && $(MAKE) --jobs=${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL=1 .. && $(MAKE) --jobs=${JOBS} )

debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && $(MAKE) --jobs=${JOBS} )

all: library

demo: library
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 .. && $(MAKE) --jobs=${JOBS} )
	( cd build && ./umfpack_simple )
	- ( cd build && ./umfpack_di_demo > my_umfpack_di_demo.out && diff ../Demo/umfpack_di_demo.out my_umfpack_di_demo.out )
	- ( cd build && ./umfpack_dl_demo > my_umfpack_dl_demo.out && diff ../Demo/umfpack_dl_demo.out my_umfpack_dl_demo.out )
	- ( cd build && ./umfpack_zi_demo > my_umfpack_zi_demo.out && diff ../Demo/umfpack_zi_demo.out my_umfpack_zi_demo.out )
	- ( cd build && ./umfpack_zl_demo > my_umfpack_zl_demo.out && diff ../Demo/umfpack_zl_demo.out my_umfpack_zl_demo.out )

# demos requiring a Fortran compiler
fdemo: demo
	( cd build && cmake $(CMAKE_OPTIONS) -DDEMO=1 -DFDEMO=1 .. && $(MAKE) --jobs=${JOBS} )
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
	( cd build && ./umf4hb         < ../Demo/HB/west0067.rua > my_umf4hb.out)
	- diff ./Demo/umf4hb.out  ./build/my_umf4hb.out 
	( cd build && ./umf4zhb        < ../Demo/HB/qc324.cua > my_umf4zhb.out  )
	- diff ./Demo/umf4zhb.out ./build/my_umf4zhb.out

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
	- $(RM) -rf build/* Config/*.tmp

purge: clean

distclean: clean

