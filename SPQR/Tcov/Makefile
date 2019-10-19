#===============================================================================
# SuiteSparseQR/Tcov/Makefile
#===============================================================================

# run statement-coverage test without Valgrind
default: go

# to run with Valgrind as well
valgrind: vgo

ccode: all

include ../../UFconfig/UFconfig.mk

# without SuiteSparseQR_expert.cpp functions:
CFLAGS = -DNEXPERT \
	-O0 -g -fprofile-arcs -ftest-coverage \
	-Wall -W -Wshadow -Winline -Wno-unused-parameter \
	-Wredundant-decls -Wdisabled-optimization \
	-ansi -fexceptions

# for statement coverage (with gcov; see go) and picky compiler warnings
CFLAGS = \
	-O0 -g -fprofile-arcs -ftest-coverage \
	-Wall -W -Wshadow -Winline -Wno-unused-parameter \
	-Wredundant-decls -Wdisabled-optimization \
	-ansi -fexceptions

# Use the standard reference BLAS, and LAPACK, both compiled with -g,
# for best diagnostics.  Also, some BLAS (such as the Goto BLAS) cause
# problems for Valgrind (using instructions that Valgrind doesn't understand).
FLIB = -lgfortran -llapack_plain -lblas_plain -lg2c

# This can be used, but severe valgrind failures may occur in the BLAS,
# because of how the BLAS is optimized:
# FLIB = $(LAPACK) $(BLAS)

CLIB = ../../CHOLMOD/Lib/libcholmod.a ../../AMD/Lib/libamd.a \
	../../COLAMD/Lib/libcolamd.a $(METIS) \
	../../CCOLAMD/Lib/libccolamd.a ../../CAMD/Lib/libcamd.a -lm

all: qrtest

library: qrtest

purge: distclean

distclean: clean
	- $(RM) qrtest qrtest_out.txt pfile tfile cov.out

clean:
	- $(RM) $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp Makefile

OBJ = \
    spqr_rmap.o \
    SuiteSparseQR_C.o \
    SuiteSparseQR_expert.o \
    spqr_parallel.o \
    spqr_kernel.o \
    spqr_analyze.o \
    spqr_assemble.o \
    spqr_cpack.o \
    spqr_csize.o \
    spqr_fcsize.o \
    spqr_debug.o \
    spqr_front.o \
    spqr_factorize.o \
    spqr_freenum.o \
    spqr_freesym.o \
    spqr_freefac.o \
    spqr_fsize.o \
    spqr_maxcolnorm.o \
    spqr_rconvert.o \
    spqr_rcount.o \
    spqr_rhpack.o \
    spqr_rsolve.o \
    spqr_stranspose1.o \
    spqr_stranspose2.o \
    spqr_hpinv.o \
    spqr_1fixed.o \
    spqr_1colamd.o \
    SuiteSparseQR.o \
    spqr_1factor.o \
    spqr_cumsum.o \
    spqr_shift.o \
    spqr_happly.o \
    spqr_panel.o \
    spqr_happly_work.o \
    SuiteSparseQR_qmult.o \
    spqr_trapezoidal.o \
    spqr_larftb.o \
    spqr_append.o \
    spqr_type.o \
    spqr_tol.o \
    qrtestc.o

$(OBJ): $(INC)

I = -I../../CHOLMOD/Include -I../../UFconfig -I../Include

C = $(CPLUSPLUS) $(CFLAGS) $(I) $(SPQR_CONFIG)

LIBS = $(CLIB) $(FLIB) $(RTLIB)

qrtestc.o: qrtestc.c $(INC)
	$(CC) $(CFLAGS) $(I) -c $<

qrtest: qrtest.cpp $(INC) $(OBJ)
	$(C) qrtest.cpp -o qrtest $(OBJ) $(LIBS)

go: qrtest
	- ./qrtest matrixlist.txt > qrtest_out.txt
	- ./cov

go1: qrtest
	- ./qrtest matrix1.txt > qrtest_out.txt
	- ./cov

vgo1: qrtest
	- valgrind --leak-check=full ./qrtest matrix1.txt > qrtest_out.txt
	- ./cov

vgo: qrtest
	- valgrind --leak-check=full ./qrtest matrixlist.txt > qrtest_out.txt
	- ./cov

spqr_1colamd.o: ../Source/spqr_1colamd.cpp
	$(C) -c $<

spqr_1factor.o: ../Source/spqr_1factor.cpp
	$(C) -c $<

spqr_1fixed.o: ../Source/spqr_1fixed.cpp
	$(C) -c $<

spqr_analyze.o: ../Source/spqr_analyze.cpp
	$(C) -c $<

spqr_parallel.o: ../Source/spqr_parallel.cpp
	$(C) -c $<

spqr_kernel.o: ../Source/spqr_kernel.cpp
	$(C) -c $<

spqr_append.o: ../Source/spqr_append.cpp
	$(C) -c $<

spqr_assemble.o: ../Source/spqr_assemble.cpp
	$(C) -c $<

spqr_cpack.o: ../Source/spqr_cpack.cpp
	$(C) -c $<

spqr_csize.o: ../Source/spqr_csize.cpp
	$(C) -c $<

spqr_cumsum.o: ../Source/spqr_cumsum.cpp
	$(C) -c $<

spqr_debug.o: ../Source/spqr_debug.cpp
	$(C) -c $<

spqr_factorize.o: ../Source/spqr_factorize.cpp
	$(C) -c $<

spqr_fcsize.o: ../Source/spqr_fcsize.cpp
	$(C) -c $<

spqr_freefac.o: ../Source/spqr_freefac.cpp
	$(C) -c $<

spqr_freenum.o: ../Source/spqr_freenum.cpp
	$(C) -c $<

spqr_freesym.o: ../Source/spqr_freesym.cpp
	$(C) -c $<

spqr_fsize.o: ../Source/spqr_fsize.cpp
	$(C) -c $<

spqr_happly.o: ../Source/spqr_happly.cpp
	$(C) -c $<

spqr_panel.o: ../Source/spqr_panel.cpp
	$(C) -c $<

spqr_happly_work.o: ../Source/spqr_happly_work.cpp
	$(C) -c $<

spqr_hpinv.o: ../Source/spqr_hpinv.cpp
	$(C) -c $<

spqr_larftb.o: ../Source/spqr_larftb.cpp
	$(C) -c $<

spqr_rconvert.o: ../Source/spqr_rconvert.cpp
	$(C) -c $<

spqr_rcount.o: ../Source/spqr_rcount.cpp
	$(C) -c $<

spqr_rhpack.o: ../Source/spqr_rhpack.cpp
	$(C) -c $<

spqr_rsolve.o: ../Source/spqr_rsolve.cpp
	$(C) -c $<

spqr_shift.o: ../Source/spqr_shift.cpp
	$(C) -c $<

spqr_stranspose1.o: ../Source/spqr_stranspose1.cpp
	$(C) -c $<

spqr_stranspose2.o: ../Source/spqr_stranspose2.cpp
	$(C) -c $<

spqr_trapezoidal.o: ../Source/spqr_trapezoidal.cpp
	$(C) -c $<

spqr_type.o: ../Source/spqr_type.cpp
	$(C) -c $<

spqr_front.o: ../Source/spqr_front.cpp
	$(C) -c $<

SuiteSparseQR_expert.o: ../Source/SuiteSparseQR_expert.cpp
	$(C) -c $<

spqr_maxcolnorm.o: ../Source/spqr_maxcolnorm.cpp
	$(C) -c $<

SuiteSparseQR_qmult.o: ../Source/SuiteSparseQR_qmult.cpp
	$(C) -c $<

SuiteSparseQR.o: ../Source/SuiteSparseQR.cpp
	$(C) -c $<

spqr_tol.o: ../Source/spqr_tol.cpp
	$(C) -c $<

SuiteSparseQR_C.o: ../Source/SuiteSparseQR_C.cpp
	$(C) -c $<

spqr_rmap.o: ../Source/spqr_rmap.cpp
	$(C) -c $<
