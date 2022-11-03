#-------------------------------------------------------------------------------
# SPEX/SPEX/SPEX_Left_LU/Tcov/Makefile: compile and run SPEX test coverage
#-------------------------------------------------------------------------------

# SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#-------------------------------------------------------------------------------

default: run

SUITESPARSE ?= $(realpath $(CURDIR)/../../..)

################################################################################

    INSTALL_LIB ?= $(SUITESPARSE)/lib

    # use 8 jobs by default
    JOBS ?= 8

    LDFLAGS = --coverage
    LDFLAGS += -L$(INSTALL_LIB)
    LDFLAGS += -Wl,-rpath=$(INSTALL_LIB)

    # remove object files, but keep compiled libraries via 'make clean'
    CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d \
        *.gcda *.gcno *.aux *.bbl *.blg *.log *.toc *.dvi *.lof *.lot

    # also remove compiled libraries, via 'make distclean'
    PURGE = *.so* *.a *.dll *.dylib *.dSYM

################################################################################

# Linux test coverage (gcc is required for test coverage)
CC = gcc
CFLAGS = -g -fprofile-arcs -ftest-coverage \
        -Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
        -Wredundant-decls -Wnested-externs -Wdisabled-optimization -std=c99 \
        -Wno-unused-parameter -I../../Include -I../Source -I../Demo \
	-I../../SPEX_Util/Include/ -I../../SPEX_Util/Source/ \
        -I../../../SuiteSparse_config -I../../../COLAMD/Include \
	-I../../../AMD/Include \
        -DSPEX_GMP_LIST_INIT=2

LDLIBS += -lmpfr -lgmp -lcolamd -lamd -lsuitesparseconfig -lm 

# run all statement coverage tests, and then check for 100% coverage
run: runtests
	./covall

all: test SPEXLLU

CS = \
    SPEX_calloc.o \
    SPEX_matrix_div.o \
    spex_cast_array.o \
    spex_cast_matrix.o \
    SPEX_check_solution.o \
    SPEX_create_default_options.o \
    spex_create_mpfr_array.o \
    spex_create_mpq_array.o \
    spex_create_mpz_array.o \
    SPEX_cumsum.o \
    spex_expand_double_array.o \
    spex_expand_mpfr_array.o \
    spex_expand_mpq_array.o \
    SPEX_finalize.o \
    SPEX_free.o \
    SPEX_gmp.o \
    SPEX_initialize.o \
    SPEX_initialize_expert.o \
    SPEX_malloc.o \
    SPEX_matrix_allocate.o \
    SPEX_matrix_check.o \
    SPEX_matrix_copy.o \
    SPEX_matrix_free.o \
    SPEX_matrix_mul.o \
    SPEX_matrix_nnz.o \
    SPEX_realloc.o \
    spex_sparse_collapse.o \
    spex_sparse_realloc.o \
    SPEX_LU_analysis_free.o \
    SPEX_LU_analyze.o \
    tcov_malloc_test.o \
    SPEX_Left_LU_backslash.o \
    spex_left_lu_back_sub.o \
    spex_left_lu_dfs.o \
    SPEX_Left_LU_factorize.o \
    spex_left_lu_forward_sub.o \
    spex_left_lu_get_largest_pivot.o \
    spex_left_lu_get_nonzero_pivot.o \
    spex_left_lu_get_pivot.o \
    spex_left_lu_get_smallest_pivot.o \
    spex_left_lu_permute_b.o \
    spex_left_lu_permute_x.o \
    spex_left_lu_reach.o \
    spex_left_lu_ref_triangular_solve.o \
    SPEX_Left_LU_solve.o 
    #SPEX_determine_symmetry.o SPEX_transpose.o \

$(CS): ../../Include/SPEX.h ../Source/spex_left_lu_internal.h ../../SPEX_Util/Source/spex_util_internal.h tcov_malloc_test.h ../Demo/demos.h

.PRECIOUS: SPEX_Left_LU_%.c spex_left_lu_%.c spexlu_demo.c demos.c SPEX_%.c spex_%.c

SPEX_Left_LU_%.c:
	- ln -s ../Source/$@

spex_left_lu_%.c:
	- ln -s ../Source/$@

spexlu_demo.c:
	- ln -s ../Demo/$@

demos.c:
	- ln -s ../Demo/$@

SPEX_%.c:
	- ln -s ../../SPEX_Util/Source/$@

spex_%.c:
	- ln -s ../../SPEX_Util/Source/$@

test: $(CS) tcov_test.c demos.c
	$(CC) $(LDFLAGS) tcov_test.c demos.c $(CFLAGS) -o tcov_test $(CS) $(LDLIBS)

SPEXLLU: $(CS) spexlu_demo.c demos.c
	$(CC) $(LDFLAGS) spexlu_demo.c demos.c $(CFLAGS) -o spexlu_demo $(CS) $(LDLIBS)

# run all statement coverage tests
runtests: all
	- ./spexlu_demo p 2 q 0
	- ./spexlu_demo p 3 q 1 o 1
	- ./spexlu_demo p 4 q 2 o 1 f ../ExampleMats/test_mat.txt ../ExampleMats/test_rhs.txt
	- ./spexlu_demo p 5
	- ./tcov_test
	- ./tcov_test 0 1 1

# To run with valgrind:
V = valgrind #--leak-check=full

# run all statement coverage tests but with valgrind
vtests: all
	- $(V) ./spexlu_demo p 2 q 0
	- $(V) ./spexlu_demo p 3 q 1 o 1
	- $(V) ./spexlu_demo p 4 q 2 o 1 f ../ExampleMats/test_mat.txt ../ExampleMats/test_rhs.txt
	- $(V) ./spexlu_demo p 5
	- $(V) ./tcov_test
	- $(V) ./tcov_test 0 1 1

# remove all files not in the original distribution, including symbolic links
clean:
	- $(RM) *.o *.bbg *.da *.gcov *.gcda *gcno
	- $(RM) tcov_test spexlu_demo *.out *.a cov.sort out
	- $(RM) -r SPEX_*.c spex_*.c *.dSYM spexlu_demo.c demos.c

purge: distclean

distclean: clean

