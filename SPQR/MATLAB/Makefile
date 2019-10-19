# ==============================================================================
# === SPQR/MATLAB/Makefile =====================================================
# ==============================================================================

# See also spqr_make.m, for a more portable way of compiling SuiteSparseQR
# for MATLAB.  Note that the mexFunctions do not make use of the C wrappers
# in SuiteSparseQR_C.cpp.

include ../../UFconfig/UFconfig.mk

RM = rm -f

OBJ = \
    spqr_rmap.o \
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
    spqr_tol.o


QR = \
    spqr_rmap.o \
    SuiteSparseQR_expert.o \
    spqr_parallel.o \
    spqr_kernel.o \
    spqr_analyze.o \
    spqr_assemble.o \
    spqr_cpack.o \
    spqr_csize.o \
    spqr_fcsize.o \
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
    spqr_panel.o \
    spqr_trapezoidal.o \
    spqr_larftb.o \
    spqr_append.o \
    spqr_type.o \
    spqr_tol.o \

QMULT = \
    spqr_happly.o \
    spqr_panel.o \
    spqr_happly_work.o \
    SuiteSparseQR_qmult.o \
    spqr_larftb.o \
    spqr_append.o \
    spqr_type.o

SING = \
    spqr_maxcolnorm.o \
    spqr_1colamd.o \
    spqr_type.o \
    spqr_tol.o \

UTIL = spqr_debug.o spqr_mx.o spqr_mx_error.o

INC = Makefile ../Include/spqr.hpp \
	../Include/SuiteSparseQR_definitions.h ../Include/SuiteSparseQR.hpp

I = -I../../CHOLMOD/Include -I../../UFconfig -I../Include

CLIB = ../../CHOLMOD/Lib/libcholmod.a ../../AMD/Lib/libamd.a \
    ../../COLAMD/Lib/libcolamd.a $(METIS) \
    ../../CCOLAMD/Lib/libccolamd.a ../../CAMD/Lib/libcamd.a -lm

all: spqr_solve.mexglx spqr.mexglx spqr_qmult.mexglx spqr_singletons.mexglx

$(OBJ): $(INC)

# R2008a:
BLAS = -lmwlapack -lmwblas
# R2006b:
# BLAS = -lmwlapack

LIBS = $(CLIB) $(BLAS) $(TBB) $(RTLIB)

MX = $(MEX) $(I) $(SPQR_CONFIG)

#-------------------------------------------------------------------------------

spqr.mexglx: spqr.cpp $(QR) $(UTIL) $(INC)
	$(MX) spqr.cpp $(QR) $(UTIL) $(LIBS)

spqr_qmult.mexglx: spqr_qmult.cpp $(QMULT) $(INC) $(UTIL)
	$(MX) spqr_qmult.cpp $(QMULT) $(UTIL) $(LIBS)

spqr_solve.mexglx: spqr_solve.cpp $(QR) $(UTIL) $(INC)
	$(MX) spqr_solve.cpp $(QR) $(UTIL) $(LIBS)

spqr_singletons.mexglx: spqr_singletons.cpp $(SING) $(UTIL) $(INC)
	$(MX) spqr_singletons.cpp $(SING) $(UTIL) $(LIBS)

#-------------------------------------------------------------------------------

spqr_analyze.o: ../Source/spqr_analyze.cpp $(INC)
	$(MX) -c $<

spqr_assemble.o: ../Source/spqr_assemble.cpp $(INC)
	$(MX) -c $<

spqr_cpack.o: ../Source/spqr_cpack.cpp $(INC)
	$(MX) -c $<

spqr_csize.o: ../Source/spqr_csize.cpp $(INC)
	$(MX) -c $<

spqr_fcsize.o: ../Source/spqr_fcsize.cpp $(INC)
	$(MX) -c $<

spqr_debug.o: ../Source/spqr_debug.cpp $(INC)
	$(MX) -c $<

spqr_factorize.o: ../Source/spqr_factorize.cpp $(INC)
	$(MX) -c $<

spqr_parallel.o: ../Source/spqr_parallel.cpp $(INC)
	$(MX) -v -c $<

spqr_kernel.o: ../Source/spqr_kernel.cpp $(INC)
	$(MX) -c $<

spqr_freenum.o: ../Source/spqr_freenum.cpp $(INC)
	$(MX) -c $<

spqr_freesym.o: ../Source/spqr_freesym.cpp $(INC)
	$(MX) -c $<

spqr_freefac.o: ../Source/spqr_freefac.cpp $(INC)
	$(MX) -c $<

spqr_fsize.o: ../Source/spqr_fsize.cpp $(INC)
	$(MX) -c $<

spqr_rconvert.o: ../Source/spqr_rconvert.cpp $(INC)
	$(MX) -c $<

spqr_rcount.o: ../Source/spqr_rcount.cpp $(INC)
	$(MX) -c $<

spqr_rhpack.o: ../Source/spqr_rhpack.cpp $(INC)
	$(MX) -c $<

spqr_rsolve.o: ../Source/spqr_rsolve.cpp $(INC)
	$(MX) -c $<

spqr_stranspose1.o: ../Source/spqr_stranspose1.cpp $(INC)
	$(MX) -c $<

spqr_stranspose2.o: ../Source/spqr_stranspose2.cpp $(INC)
	$(MX) -c $<

spqr_hpinv.o: ../Source/spqr_hpinv.cpp $(INC)
	$(MX) -c $<

spqr_1fixed.o: ../Source/spqr_1fixed.cpp $(INC)
	$(MX) -c $<

spqr_1colamd.o: ../Source/spqr_1colamd.cpp $(INC)
	$(MX) -c $<

spqr_1factor.o: ../Source/spqr_1factor.cpp $(INC)
	$(MX) -c $<

spqr_cumsum.o: ../Source/spqr_cumsum.cpp $(INC)
	$(MX) -c $<

spqr_shift.o: ../Source/spqr_shift.cpp $(INC)
	$(MX) -c $<

spqr_happly.o: ../Source/spqr_happly.cpp $(INC)
	$(MX) -c $<

spqr_panel.o: ../Source/spqr_panel.cpp $(INC)
	$(MX) -c $<

spqr_happly_work.o: ../Source/spqr_happly_work.cpp $(INC)
	$(MX) -c $<

spqr_trapezoidal.o: ../Source/spqr_trapezoidal.cpp $(INC)
	$(MX) -c $<

spqr_larftb.o: ../Source/spqr_larftb.cpp $(INC)
	$(MX) -c $<

spqr_append.o: ../Source/spqr_append.cpp $(INC)
	$(MX) -c $<

spqr_type.o: ../Source/spqr_type.cpp $(INC)
	$(MX) -c $<

spqr_maxcolnorm.o: ../Source/spqr_maxcolnorm.cpp $(INC)
	$(MX) -c $<

spqr_front.o: ../Source/spqr_front.cpp $(INC)
	$(MX) -c $<

SuiteSparseQR.o: ../Source/SuiteSparseQR.cpp $(INC)
	$(MX) -c $<

SuiteSparseQR_qmult.o: ../Source/SuiteSparseQR_qmult.cpp $(INC)
	$(MX) -c $<

spqr_tol.o: ../Source/spqr_tol.cpp $(INC)
	$(MX) -c $<

SuiteSparseQR_expert.o: ../Source/SuiteSparseQR_expert.cpp $(INC)
	$(MX) -c $<

spqr_rmap.o: ../Source/spqr_rmap.cpp $(INC)
	$(MX) -c $<

#-------------------------------------------------------------------------------

spqr_mx.o: spqr_mx.cpp $(INC)
	$(MX) -c $<

spqr_mx_error.o: spqr_mx_error.c $(INC)
	$(MX) -c $<

#-------------------------------------------------------------------------------

purge:
	$(RM) *.o rename.h *.mex* pfile tfile
