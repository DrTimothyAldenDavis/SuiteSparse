#-------------------------------------------------------------------------------
# SLIP_LU/Tcov/Makefile: compile and run SLIP_LU test coverage
#-------------------------------------------------------------------------------

# SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
# Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
# SLIP_LU/License for the license.

#-------------------------------------------------------------------------------

default: run

include ../../SuiteSparse_config/SuiteSparse_config.mk

# Linux test coverage
CC = gcc
CFLAGS = -g -fprofile-arcs -ftest-coverage \
        -Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
        -Wredundant-decls -Wnested-externs -Wdisabled-optimization -std=c99 \
        -Wno-unused-parameter -I../Include -I../Source -I../Demo \
        -I../../SuiteSparse_config -I../../COLAMD/Include -I../../AMD/Include \
        -DSLIP_GMP_LIST_INIT=2 -DSLIP_LU_TCOV

LDLIBS += -lm -lgmp -lmpfr -lcolamd -lamd -lsuitesparseconfig

# run all statement coverage tests, and then check for 100% coverage
run: runtests
	./covall

all: test SLIPLU

CS = \
    slip_matrix_div.o slip_matrix_mul.o slip_back_sub.o \
    SLIP_backslash.o \
    slip_cast_array.o \
    slip_cast_matrix.o \
    slip_check_solution.o \
    SLIP_create_default_options.o \
    slip_create_mpfr_array.o \
    slip_create_mpq_array.o \
    slip_create_mpz_array.o \
    slip_cumsum.o \
    slip_dfs.o \
    slip_expand_double_array.o \
    slip_expand_mpfr_array.o \
    slip_expand_mpq_array.o \
    SLIP_finalize.o \
    slip_forward_sub.o \
    SLIP_malloc.o \
    SLIP_calloc.o \
    SLIP_realloc.o \
    SLIP_free.o \
    slip_get_largest_pivot.o \
    slip_get_nonzero_pivot.o \
    slip_get_pivot.o \
    slip_get_smallest_pivot.o \
    SLIP_gmp.o \
    SLIP_initialize.o \
    SLIP_initialize_expert.o \
    SLIP_LU_analysis_free.o \
    SLIP_LU_analyze.o \
    SLIP_LU_factorize.o \
    SLIP_LU_solve.o \
    SLIP_matrix_allocate.o \
    SLIP_matrix_check.o \
    SLIP_matrix_copy.o \
    SLIP_matrix_free.o \
    SLIP_matrix_nnz.o \
    slip_permute_b.o \
    slip_permute_x.o \
    slip_reach.o \
    slip_ref_triangular_solve.o \
    slip_sparse_collapse.o \
    slip_sparse_realloc.o \
    tcov_malloc_test.o

$(CS): ../Include/SLIP_LU.h ../Source/slip_internal.h tcov_malloc_test.h ../Demo/demos.h

.PRECIOUS: SLIP_%.c slip_%.c SLIPLU.c demos.c

SLIP_%.c:
	- ln -s ../Source/$@

slip_%.c:
	- ln -s ../Source/$@

SLIPLU.c:
	- ln -s ../Demo/$@

demos.c:
	- ln -s ../Demo/$@

test: $(CS) tcov_test.c demos.c
	$(CC) $(LDFLAGS) tcov_test.c demos.c $(CFLAGS) -o tcov_test $(CS) $(LDLIBS)

SLIPLU: $(CS) SLIPLU.c demos.c
	$(CC) $(LDFLAGS) SLIPLU.c demos.c $(CFLAGS) -o SLIPLU $(CS) $(LDLIBS)

# run all statement coverage tests
runtests: all
	- ./SLIPLU p 2 q 0
	- ./SLIPLU p 3 q 1 o 1
	- ./SLIPLU p 4 q 2 o 1 f ../ExampleMats/test_mat.txt ../ExampleMats/test_rhs.txt
	- ./SLIPLU p 5
	- ./tcov_test
	- ./tcov_test 0 1 1

# To run with valgrind:
V = valgrind #--leak-check=full

# run all statement coverage tests but with valgrind
vtests: all
	- $(V) ./SLIPLU p 2 q 0
	- $(V) ./SLIPLU p 3 q 1 o 1
	- $(V) ./SLIPLU p 4 q 2 o 1 f ../ExampleMats/test_mat.txt ../ExampleMats/test_rhs.txt
	- $(V) ./SLIPLU p 5
	- $(V) ./tcov_test
	- $(V) ./tcov_test 0 1 1

# remove all files not in the original distribution, including symbolic links
clean:
	- $(RM) *.o *.bbg *.da *.gcov *.gcda *gcno
	- $(RM) tcov_test SLIPLU *.out *.a cov.sort out
	- $(RM) -r SLIP_*.c slip_*.c *.dSYM SLIPLU.c demos.c

purge: distclean

distclean: clean

