#===============================================================================
# ParU/Tcov/Makefile
#===============================================================================

# ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
# All Rights Reserved.
# SPDX-License-Identifier: GPL-3.0-or-later

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
	-I../../CHOLMOD/Include -I../../AMD/Include   -I../Include -I../Source \
        -I/usr/local/cuda/include

C = $(CXX) $(CF) $(I) -Wno-write-strings -std=c++11 \
    -DPARU_COVERAGE -DPARU_ALLOC_TESTING -DBLAS32
#C2 = $(CC) $(CF) $(I) -Wno-write-strings -std=c11

all: paru_quick_test paru_brutal_test paru_c_test


#-------------------------------------------------------------------------------
INC = ../Include/ParU.h

OBJ = \
	paru_mem.o \
	paru_nthreads.o \
	paru_free_work.o \
	ParU_FreeSymbolic.o \
	ParU_FreeNumeric.o \
	ParU_InitControl.o \
	ParU_FreeControl.o \
	paru_memset.o \
	paru_memcpy.o \
	paru_heap.o \
	paru_hash.o \
	paru_pivotal.o \
	ParU_Analyze.o \
	paru_init_rowFronts.o\
	paru_tuples.o\
	paru_front.o\
	ParU_Factorize.o\
	paru_exec_tasks.o\
	paru_fs_factorize.o\
	paru_create_element.o\
	paru_assemble_row2U.o\
	paru_dtrsm.o\
	paru_dgemm.o\
	paru_init_rel.o\
	paru_update_rowDeg.o\
	paru_update_rel_ind.o\
	paru_cumsum.o\
	paru_intersection.o\
	paru_bin_search.o\
	paru_assemble.o\
	paru_prior_assemble.o \
	ParU_Perm.o \
	ParU_InvPerm.o \
	paru_finalize_perm.o \
	ParU_LSolve.o \
	ParU_USolve.o \
	ParU_Solve.o \
	ParU_Residual.o \
	paru_backward.o \
	paru_gaxpy.o \
	paru_norms.o \
	paru_diag_update.o \
	paru_full_summed.o \
	paru_tasked_dgemm.o \
	paru_tasked_dtrsm.o \
	ParU_C.o \
	ParU_Get.o \
	ParU_Set.o \
	ParU_Version.o  \
	paru_umfpack_info.o  \
	SuiteSparse_config.o

# paru_print and paru_write are used only in debug mode:
#	paru_print.o \
#	paru_write.o

$(OBJ): $(INC)

# VFLAGS = --leak-check=yes  --log-file="valgrindlogfile.log"\
# 		   --show-leak-kinds=all --track-origins=yes

VFLAGS = --leak-check=yes --show-leak-kinds=all --track-origins=yes

# V = valgrind $(VFLAGS)
V =

#-------------------------------------------------------------------------------
paru_brutal_test: paru_brutal_test.cpp $(INC) $(OBJ)
	$(C)  paru_brutal_test.cpp -o paru_brutal_test $(OBJ) $(LIBS)

paru_quick_test: paru_quick_test.cpp $(INC) $(OBJ)
	$(C)  paru_quick_test.cpp -o paru_quick_test $(OBJ) $(LIBS)

paru_c_test: paru_c_test.cpp $(INC) $(OBJ)
	$(C)  paru_c_test.cpp -o paru_c_test $(OBJ) $(LIBS)

go: paru_brutal_test paru_quick_test paru_c_test
	$(V) ./paru_c_test < ../Matrix/west0067.mtx
	- ./cov
	$(V) ./paru_quick_test -12 3 0 < ../Matrix/bp_1200.mtx
	$(V) ./paru_quick_test -12 3 1 < ../Matrix/bp_1200.mtx
	$(V) ./paru_quick_test -12 3 2 < ../Matrix/bp_1200.mtx
	$(V) ./paru_quick_test -12 3 < ../Matrix/bp_1200.mtx
	$(V) ./paru_quick_test -12 1 0 < ../Matrix/reorientation_1.mtx
	$(V) ./paru_quick_test -12 1 1 < ../Matrix/reorientation_1.mtx
	$(V) ./paru_quick_test -12 1 2 < ../Matrix/reorientation_1.mtx
	$(V) ./paru_quick_test -12 1 < ../Matrix/reorientation_1.mtx
	$(V) ./paru_quick_test -12 1 0 < ../Matrix/watt_2.mtx
	$(V) ./paru_quick_test -12 1 1 < ../Matrix/watt_2.mtx
	$(V) ./paru_quick_test -12 1 2 < ../Matrix/watt_2.mtx
	$(V) ./paru_quick_test -12 1 < ../Matrix/watt_2.mtx
	- ./cov
	$(V) ./paru_quick_test < ../Matrix/tumorAntiAngiogenesis_2.mtx
	$(V) ./paru_quick_test < ../Matrix/olm500.mtx
	- ./cov
	$(V) ./paru_quick_test < ../Matrix/adder_dcop_05.mtx
	$(V) ./paru_quick_test < ../Matrix/bayer10.mtx
	$(V) ./paru_quick_test 104 < ../Matrix/rajat01.mtx
	$(V) ./paru_quick_test < ../Matrix/rajat19.mtx
	$(V) ./paru_brutal_test < ../Matrix/west0497.mtx
	$(V) ./paru_quick_test 104 < ../Matrix/gent113.mtx
	- ./cov
	$(V) ./paru_quick_test < ../Matrix/hangGlider_2.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/Tina_AskCal.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/GD98_a.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/a0.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/a1.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/Tina_AskCal_perm.mtx
	- ./cov
	$(V) ./paru_brutal_test 104 < ../Matrix/ash219.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/lp_e226.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/r2.mtx
	$(V) ./paru_brutal_test < ../Matrix/LFAT5.mtx
	$(V) ./paru_quick_test < ../Matrix/arrow.mtx
	- ./cov
	$(V) ./paru_brutal_test < ../Matrix/a2.mtx
	$(V) ./paru_brutal_test < ../Matrix/az88.mtx
	$(V) ./paru_quick_test 102 < ../Matrix/young1c.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/s32.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/lp_share1b.mtx
	- ./cov
	$(V) ./paru_brutal_test -14 < ../Matrix/cage3.mtx
	$(V) ./paru_brutal_test -15 < ../Matrix/b1_ss.mtx
	$(V) ./paru_brutal_test -15 < ../Matrix/lfat5b.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/c32.mtx
	$(V) ./paru_brutal_test -15 < ../Matrix/bfwa62.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/Ragusa16.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/lp_e226_transposed.mtx
	- ./cov
	$(V) ./paru_brutal_test 105 < ../Matrix/row0.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/Groebner_id2003_aug.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/c2.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/a4.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/problem.mtx
	$(V) ./paru_brutal_test -14 < ../Matrix/permuted_b1_ss.mtx
	$(V) ./paru_brutal_test < ../Matrix/pwr01b.mtx
	- ./cov
	$(V) ./paru_brutal_test 104 < ../Matrix/Franz6_id1959_aug.mtx
	$(V) ./paru_brutal_test 104 < ../Matrix/Ragusa16_pattern.mtx
	$(V) ./paru_brutal_test < ../Matrix/temp.mtx
	$(V) ./paru_brutal_test -15 < ../Matrix/cage5.mtx
	$(V) ./paru_c_test 104 < ../Matrix/a04.mtx
	$(V) ./paru_c_test 104 < ../Matrix/lpi_galenet.mtx
	$(V) ./paru_quick_test < ../Matrix/nnc1374.mtx
	- ./cov
	$(V) ./paru_brutal_test < ../Matrix/494_bus.mtx
	- ./cov
	$(V) ./paru_quick_test < ../Matrix/c-62.mtx
	- ./cov
	$(V) ./paru_quick_test < ../Matrix/c-62.mtx
	- ./cov

vgo: paru_brutal_test
	- valgrind --leak-check=yes --track-origins=yes -s ./paru_brutal_test \
		< ../Demo/Matrix/b1_ss.mtx > parutest_out.txt
	- ./cov

qgo:
	- ./paru_quick_test < ../Demo/Matrix/b1_ss.mtx > parutest_out.txt
	- ./cov

#-------------------------------------------------------------------------------
paru_mem.o: ../Source/paru_mem.cpp
	$(C) -c $<

paru_nthreads.o: ../Source/paru_nthreads.cpp
	$(C) -c $<

paru_free_work.o: ../Source/paru_free_work.cpp
	$(C) -c $<

ParU_FreeSymbolic.o: ../Source/ParU_FreeSymbolic.cpp
	$(C) -c $<

ParU_FreeNumeric.o: ../Source/ParU_FreeNumeric.cpp
	$(C) -c $<

ParU_FreeControl.o: ../Source/ParU_FreeControl.cpp
	$(C) -c $<

ParU_InitControl.o: ../Source/ParU_InitControl.cpp
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

ParU_Analyze.o: ../Source/ParU_Analyze.cpp
	$(C) -c $<

paru_init_rowFronts.o: ../Source/paru_init_rowFronts.cpp
	$(C) -c $<

paru_tuples.o: ../Source/paru_tuples.cpp
	$(C) -c $<

paru_front.o: ../Source/paru_front.cpp
	$(C) -c $<

ParU_Factorize.o: ../Source/ParU_Factorize.cpp
	$(C) -c $<

paru_exec_tasks.o: ../Source/paru_exec_tasks.cpp
	$(C) -c $<

paru_fs_factorize.o: ../Source/paru_fs_factorize.cpp
	$(C) -c $<

paru_create_element.o: ../Source/paru_create_element.cpp
	$(C) -c $<

paru_assemble_row2U.o: ../Source/paru_assemble_row2U.cpp
	$(C) -c $<

paru_dtrsm.o: ../Source/paru_dtrsm.cpp
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

ParU_Perm.o: ../Source/ParU_Perm.cpp
	$(C) -c $<

ParU_InvPerm.o: ../Source/ParU_InvPerm.cpp
	$(C) -c $<

paru_finalize_perm.o: ../Source/paru_finalize_perm.cpp
	$(C) -c $<

ParU_LSolve.o: ../Source/ParU_LSolve.cpp
	$(C) -c $<

ParU_USolve.o: ../Source/ParU_USolve.cpp
	$(C) -c $<

ParU_Solve.o: ../Source/ParU_Solve.cpp
	$(C) -c $<

ParU_Residual.o: ../Source/ParU_Residual.cpp
	$(C) -c $<

paru_gaxpy.o: ../Source/paru_gaxpy.cpp
	$(C) -c $<

paru_norms.o: ../Source/paru_norms.cpp
	$(C) -c $<

paru_diag_update.o: ../Source/paru_diag_update.cpp
	$(C) -c $<

paru_full_summed.o: ../Source/paru_full_summed.cpp
	$(C) -c $<

paru_tasked_dgemm.o: ../Source/paru_tasked_dgemm.cpp
	$(C) -c $<

paru_tasked_dtrsm.o: ../Source/paru_tasked_dtrsm.cpp
	$(C) -c $<

ParU_C.o: ../Source/ParU_C.cpp
	$(C) -c $<

ParU_Get.o: ../Source/ParU_Get.cpp
	$(C) -c $<

ParU_Set.o: ../Source/ParU_Set.cpp
	$(C) -c $<

paru_cp_control.o: ../Source/paru_cp_control.cpp
	$(C) -c $<

ParU_Version.o: ../Source/ParU_Version.cpp
	$(C) -c $<

paru_umfpack_info.o: ../Source/paru_umfpack_info.cpp
	$(C) -c $<

SuiteSparse_config.o: ../../SuiteSparse_config/SuiteSparse_config.c
	gcc -c -fopenmp -DBLAS32 $<

#-------------------------------------------------------------------------------

paru_backward.o: ./paru_backward.cpp
	$(C) -c $<

#-------------------------------------------------------------------------------
just: paru_brutal_test paru_quick_test paru_c_test

purge: clean

clean:
	- $(RM) cov.out parutest_out.txt *.o *.gcno  *.gcov *.gcda gmon.out
	- $(RM) paru_brutal_test paru_c_test paru_quick_test
