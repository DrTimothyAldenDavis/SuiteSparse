#===============================================================================
# ParU/Tcov/Makefile
#===============================================================================

# ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
# All Rights Reserved.
# SPDX-License-Identifier: GNU GPL 3.0

default: go

#for the tests that also use valgrind. Just for very small input matrices
valgrind: vgo

ccode: all

SUITESPARSE ?= $(realpath $(CURDIR)/../..)


#CC = gcc
CXX = g++

LIBS = -L../../lib -lumfpack -lcholmod -lamd -lcolamd -lm \
    -lcamd -lccolamd -lblas -Wl,--rpath=$(SUITESPARSE)/lib

# for statement coverage (with gcov; see go) and picky compiler warnings
CF = -pg -O0 -g -fprofile-arcs -ftest-coverage \
	-Wall -W -Wshadow -Winline -Wno-unused-parameter \
	 -Wdisabled-optimization -fexceptions -fopenmp 

I = -I../../SuiteSparse_config -I../../UMFPACK/Source -I../../UMFPACK/Include \
	-I../../CHOLMOD/Include -I../../AMD/Include   -I../Include -I../Source

C = $(CXX) $(CF) $(I) -Wno-write-strings -std=c++11 \
    -DPARU_COVERAGE -DPARU_ALLOC_TESTING -DBLAS32
#C2 = $(CC) $(CF) $(I) -Wno-write-strings -std=c11

all: paru_quicker_test paru_brutal_test paru_ctest
	

#-------------------------------------------------------------------------------
INC = ../Include/ParU.hpp 

OBJ = \
        paru_mem.o \
        paru_memset.o \
        paru_memcpy.o \
        paru_heap.o \
        paru_hash.o \
        paru_pivotal.o \
        paru_analyze.o \
	paru_init_rowFronts.o\
	paru_tuples.o\
	paru_front.o\
	paru_factorize.o\
	paru_fs_factorize.o\
	paru_create_element.o\
	paru_assemble_row2U.o\
	paru_trsm.o\
	paru_dgemm.o\
	paru_init_rel.o\
	paru_update_rowDeg.o\
	paru_update_rel_ind.o\
	paru_cumsum.o\
	paru_intersection.o\
	paru_bin_search.o\
	paru_assemble.o\
	paru_prior_assemble.o \
	paru_perm.o \
	paru_finalize_perm.o \
	paru_lsolve.o \
	paru_usolve.o \
	paru_solve.o \
	paru_residual.o \
	paru_backward.o \
	paru_gaxpy.o \
	paru_norms.o \
	paru_Diag_update.o \
	paru_full_summed.o \
	paru_tasked_dgemm.o \
	paru_tasked_trsm.o \
        paru_c.o \
	paru_version.o  \
	SuiteSparse_config.o 

#	paru_print.o \   # These methods are used only in debug mode
#	paru_write.o


$(OBJ): $(INC)

VFLAGS = --leak-check=yes  --log-file="valgrindlogfile.log"\
		   --show-leak-kinds=all --track-origins=yes 

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
paru_brutal_test: paru_brutal_test.cpp $(INC) $(OBJ) 
	$(C)  paru_brutal_test.cpp -o x_brutal_test $(OBJ) $(LIBS)

paru_quicker_test: paru_quicker_test.cpp $(INC) $(OBJ) 
	$(C)  paru_quicker_test.cpp -o x_quick_test $(OBJ) $(LIBS)

paru_ctest: paru_ctest.cpp $(INC) $(OBJ) 
	$(C)  paru_ctest.cpp -o x_paru_c $(OBJ) $(LIBS)


go: paru_brutal_test paru_quicker_test paru_ctest
#	- ./x_brutal_test < ../Demo/Matrix/b1_ss.mtx > parutest_out.txt
	- ./trigger.sh
	- ./cov

vgo: paru_brutal_test
	- valgrind --leak-check=yes --track-origins=yes -s ./x_brutal_test \
		< ../Demo/Matrix/b1_ss.mtx > parutest_out.txt
	- ./cov

qgo:
	- ./x_quick_test < ../Demo/Matrix/b1_ss.mtx > parutest_out.txt
	- ./cov

#-------------------------------------------------------------------------------
paru_mem.o: ../Source/paru_mem.cpp
	$(C) -c $<

paru_memset.o: ../Source/paru_memset.cpp
	$(C) -c $<

paru_memcpy.o: ../Source/paru_memcpy.cpp
	$(C) -c $<

paru_heap.o: ../Source/paru_heap.cpp
	$(C) -c $<

paru_hash.o: ../Source/paru_hash.cpp
	$(C) -c $<
	
paru_pivotal.o: ../Source/paru_pivotal.cpp
	$(C) -c $<

paru_analyze.o: ../Source/paru_analyze.cpp
	$(C) -c $<

paru_init_rowFronts.o: ../Source/paru_init_rowFronts.cpp
	$(C) -c $<

paru_tuples.o: ../Source/paru_tuples.cpp
	$(C) -c $<

paru_front.o: ../Source/paru_front.cpp
	$(C) -c $<

paru_factorize.o: ../Source/paru_factorize.cpp
	$(C) -c $<

paru_fs_factorize.o: ../Source/paru_fs_factorize.cpp
	$(C) -c $<

paru_create_element.o: ../Source/paru_create_element.cpp
	$(C) -c $<

paru_assemble_row2U.o: ../Source/paru_assemble_row2U.cpp
	$(C) -c $<

paru_trsm.o: ../Source/paru_trsm.cpp
	$(C) -c $<

paru_dgemm.o: ../Source/paru_dgemm.cpp
	$(C) -c $<

paru_print.o: ../Source/paru_print.cpp
	$(C) -c $<

paru_init_rel.o: ../Source/paru_init_rel.cpp
	$(C) -c $<

paru_write.o: ../Source/paru_write.cpp
	$(C) -c $<

paru_update_rowDeg.o: ../Source/paru_update_rowDeg.cpp
	$(C) -c $<

paru_update_rel_ind.o: ../Source/paru_update_rel_ind.cpp
	$(C) -c $<

paru_cumsum.o: ../Source/paru_cumsum.cpp
	$(C) -c $<

paru_intersection.o: ../Source/paru_intersection.cpp
	$(C) -c $<

paru_bin_search.o: ../Source/paru_bin_search.cpp
	$(C) -c $<

paru_prior_assemble.o:  ../Source/paru_prior_assemble.cpp
	$(C) -c $<

paru_assemble.o: ../Source/paru_assemble.cpp
	$(C) -c $<

paru_perm.o: ../Source/paru_perm.cpp
	$(C) -c $<

paru_finalize_perm.o: ../Source/paru_finalize_perm.cpp
	$(C) -c $<

paru_lsolve.o: ../Source/paru_lsolve.cpp
	$(C) -c $<

paru_usolve.o: ../Source/paru_usolve.cpp
	$(C) -c $<

paru_solve.o: ../Source/paru_solve.cpp
	$(C) -c $<

paru_residual.o: ../Source/paru_residual.cpp
	$(C) -c $<

paru_backward.o: ../Source/paru_backward.cpp
	$(C) -c $<

paru_gaxpy.o: ../Source/paru_gaxpy.cpp
	$(C) -c $<

paru_norms.o: ../Source/paru_norms.cpp
	$(C) -c $<

paru_Diag_update.o: ../Source/paru_Diag_update.cpp
	$(C) -c $<

paru_full_summed.o: ../Source/paru_full_summed.cpp
	$(C) -c $<

paru_tasked_dgemm.o: ../Source/paru_tasked_dgemm.cpp
	$(C) -c $<

paru_tasked_trsm.o: ../Source/paru_tasked_trsm.cpp
	$(C) -c $<

paru_c.o: ../Source/paru_c.cpp
	$(C) -c $<

paru_version.o: ../Source/paru_version.cpp
	$(C) -c $<

SuiteSparse_config.o: ../../SuiteSparse_config/SuiteSparse_config.c
	gcc -c -fopenmp -DBLAS32 $<

#-------------------------------------------------------------------------------
purge: clean

clean: 
	- $(RM) x_* cov.out parutest_out.txt *.o *.gcno  *.gcov *.gcda gmon.out
