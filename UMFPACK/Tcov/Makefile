#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/Tcov/Makefile
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------

# statement coverage test

# choose the test options:  also see ../Source/umf_config.h
#
# NBLAS: do not use the BLAS; use plain C code instead
# NRECIPROCAL: do not multiply by the reciprocal when scaling
# TEST_FOR_INTEGER_OVERFLOW: set the max mark to 3n, to test clearing mark
# NO_DIVIDE_BY_ZERO: if true, never divide by zero
#
# OPTION = 
# OPTION = -DNBLAS -DNRECIPROCAL
  OPTION = -DNBLAS -DNRECIPROCAL -DTEST_FOR_INTEGER_OVERFLOW
# OPTION = -DNBLAS -DNRECIPROCAL -DTEST_FOR_INTEGER_OVERFLOW -DNO_DIVIDE_BY_ZERO

CF = $(OPTION) -g --coverage -fprofile-abs-path -std=c11 -DTESTING
I = -I../../include -I../Include -I../../UMFPACK/Source -I../../AMD/Include
LIB = -Wl,--rpath=/home/faculty/d/davis/dev2/SuiteSparse/lib \
                -L/home/faculty/d/davis/dev2/SuiteSparse/lib \
      -lsuitesparseconfig -lamd -lcholmod -lblas -lm
SRC = ../Source2/*.c

SRC2 = $(notdir $(wildcard $(SRC)))

OBJ = $(SRC2:.c=.o)

.PRECIOUS: %.c

%.o: ../Source2/%.c
	- ln -s $< $(basename $@).c
	gcc $(CF) $(I) -c $(basename $@).c -o $(basename $@).o

default: $(OBJ) ut.c
	gcc -DDINT  ut.c $(CF) $(I) $(OBJ) $(LIB) -o ut_di
	gcc -DDLONG ut.c $(CF) $(I) $(OBJ) $(LIB) -o ut_dl
	gcc -DZINT  ut.c $(CF) $(I) $(OBJ) $(LIB) -o ut_zi
	gcc -DZLONG ut.c $(CF) $(I) $(OBJ) $(LIB) -o ut_zl
	./ut_di > ut_di.out
	./ut_dl > ut_dl.out
	./ut_zi > ut_zi.out
	./ut_zl > ut_zl.out
	- ./gcovs umf*.c
	- ./cover umf*v > cover.out
	- echo -n "statments not yet tested: "
	- grep -c "#####" cover.out

ut.c:
	- ln -s Source/ut.c

purge:
	$(RM) *.c *.o ut_?? *.gcno *.gcda *.gcov debug.amd *.umf A500 *.out

