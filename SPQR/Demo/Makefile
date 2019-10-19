#===============================================================================
# SuiteSparseQR/Demo/Makefile
#===============================================================================

default: all

ccode: all

include ../../UFconfig/UFconfig.mk

CLIB = ../Lib/libspqr.a \
	../../CHOLMOD/Lib/libcholmod.a ../../AMD/Lib/libamd.a \
	../../COLAMD/Lib/libcolamd.a $(METIS) \
	../../CCOLAMD/Lib/libccolamd.a ../../CAMD/Lib/libcamd.a -lm

# use the BLAS and LAPACK defined by UFconfig.mk; do not use valgrind 
FLIB = $(LAPACK) $(BLAS)
V = 

# To use Valgrind and the plain BLAS and plain LAPACK (non-optimized):
# FLIB = -lgfortran -llapack_plain -lblas_plain -lg2c
# V = valgrind --quiet

all: qrdemo qrsimple qrsimplec
	- $(V) ./qrsimple < ../Matrix/ash219.mtx
	- $(V) ./qrsimplec < ../Matrix/ash219.mtx
	- $(V) ./qrsimple < ../Matrix/west0067.mtx
	- $(V) ./qrsimplec < ../Matrix/west0067.mtx
	- $(V) ./qrdemo < ../Matrix/a2.mtx
	- $(V) ./qrdemo < ../Matrix/r2.mtx
	- $(V) ./qrdemo < ../Matrix/a04.mtx
	- $(V) ./qrdemo < ../Matrix/a2.mtx
	- $(V) ./qrdemo < ../Matrix/west0067.mtx
	- $(V) ./qrdemo < ../Matrix/c2.mtx
	- $(V) ./qrdemo < ../Matrix/a0.mtx
	- $(V) ./qrdemo < ../Matrix/lfat5b.mtx
	- $(V) ./qrdemo < ../Matrix/bfwa62.mtx
	- $(V) ./qrdemo < ../Matrix/LFAT5.mtx
	- $(V) ./qrdemo < ../Matrix/b1_ss.mtx
	- $(V) ./qrdemo < ../Matrix/bcspwr01.mtx
	- $(V) ./qrdemo < ../Matrix/lpi_galenet.mtx
	- $(V) ./qrdemo < ../Matrix/lpi_itest6.mtx
	- $(V) ./qrdemo < ../Matrix/ash219.mtx
	- $(V) ./qrdemo < ../Matrix/a4.mtx
	- $(V) ./qrdemo < ../Matrix/s32.mtx
	- $(V) ./qrdemo < ../Matrix/c32.mtx
	- $(V) ./qrdemo < ../Matrix/lp_share1b.mtx
	- $(V) ./qrdemo < ../Matrix/a1.mtx
	- $(V) ./qrdemo < ../Matrix/GD06_theory.mtx
	- $(V) ./qrdemo < ../Matrix/GD01_b.mtx
	- $(V) ./qrdemo < ../Matrix/Tina_AskCal_perm.mtx
	- $(V) ./qrdemo < ../Matrix/Tina_AskCal.mtx
	- $(V) ./qrdemo < ../Matrix/GD98_a.mtx
	- $(V) ./qrdemo < ../Matrix/Ragusa16.mtx
	- $(V) ./qrdemo < ../Matrix/young1c.mtx

cdemo: qrdemoc
	- $(V) ./qrdemoc < ../Matrix/a2.mtx
	- $(V) ./qrdemoc < ../Matrix/r2.mtx
	- $(V) ./qrdemoc < ../Matrix/a04.mtx
	- $(V) ./qrdemoc < ../Matrix/a2.mtx
	- $(V) ./qrdemoc < ../Matrix/west0067.mtx
	- $(V) ./qrdemoc < ../Matrix/c2.mtx
	- $(V) ./qrdemoc < ../Matrix/a0.mtx
	- $(V) ./qrdemoc < ../Matrix/lfat5b.mtx
	- $(V) ./qrdemoc < ../Matrix/bfwa62.mtx
	- $(V) ./qrdemoc < ../Matrix/LFAT5.mtx
	- $(V) ./qrdemoc < ../Matrix/b1_ss.mtx
	- $(V) ./qrdemoc < ../Matrix/bcspwr01.mtx
	- $(V) ./qrdemoc < ../Matrix/lpi_galenet.mtx
	- $(V) ./qrdemoc < ../Matrix/lpi_itest6.mtx
	- $(V) ./qrdemoc < ../Matrix/ash219.mtx
	- $(V) ./qrdemoc < ../Matrix/a4.mtx
	- $(V) ./qrdemoc < ../Matrix/s32.mtx
	- $(V) ./qrdemoc < ../Matrix/c32.mtx
	- $(V) ./qrdemoc < ../Matrix/lp_share1b.mtx
	- $(V) ./qrdemoc < ../Matrix/a1.mtx
	- $(V) ./qrdemoc < ../Matrix/GD06_theory.mtx
	- $(V) ./qrdemoc < ../Matrix/GD01_b.mtx
	- $(V) ./qrdemoc < ../Matrix/Tina_AskCal_perm.mtx
	- $(V) ./qrdemoc < ../Matrix/Tina_AskCal.mtx
	- $(V) ./qrdemoc < ../Matrix/GD98_a.mtx
	- $(V) ./qrdemoc < ../Matrix/Ragusa16.mtx
	- $(V) ./qrdemoc < ../Matrix/young1c.mtx

library:
	( cd ../Lib ; $(MAKE) )

purge: distclean

distclean: clean
	- $(RM) qrdemo qrdemoc qrsimple qrsimplec

clean:
	- $(RM) $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp Makefile

I = -I../../CHOLMOD/Include -I../../UFconfig -I../Include

C = $(CPLUSPLUS) $(CFLAGS) $(SPQR_CONFIG) $(I)

LIBS = $(CLIB) $(FLIB) $(TBB) $(RTLIB)

qrsimple: qrsimple.cpp library $(INC)
	$(C) qrsimple.cpp -o qrsimple $(LIBS)

qrdemo: qrdemo.cpp library $(INC)
	$(C) qrdemo.cpp -o qrdemo $(LIBS)

# compile the C code with gcc, but link with g++ to use SuiteSparseQR:

qrdemoc.o: qrdemoc.c $(INC)
	$(CC) $(CFLAGS) $(SPQR_CONFIG) -ansi $(I) -c $<

qrdemoc: qrdemoc.o library $(INC)
	$(C) -o qrdemoc qrdemoc.o $(LIBS)

qrsimplec.o: qrsimplec.c $(INC)
	$(CC) $(CFLAGS) $(SPQR_CONFIG) -ansi $(I) -c $<

qrsimplec: qrsimplec.o library $(INC)
	$(C) -o qrsimplec qrsimplec.o $(LIBS)

