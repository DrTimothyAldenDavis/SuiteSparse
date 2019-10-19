#===============================================================================
# SuiteSparseQR/Demo/Makefile
#===============================================================================

default: all

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# the optional Partition module requires METIS, CAMD, and CCOLAMD
I_WITH_PARTITION = 
LIB_WITH_PARTITION =
CONFIG = -DNPARTITION
# check if METIS is requested and available
ifeq (,$(findstring -DNPARTITION, $(CHOLMOD_CONFIG)))
    # METIS is requested.  See if it is available
    ifeq ($(METIS_PATH), $(wildcard $(METIS_PATH)))
        ifeq (../../CAMD, $(wildcard ../../CAMD))
            ifeq (../../CCOLAMD, $(wildcard ../../CCOLAMD))
                # METIS, CAMD, and CCOLAMD are available
                LIB_WITH_PARTITION = $(METIS) ../../CCOLAMD/Lib/libccolamd.a \
                    ../../CAMD/Lib/libcamd.a
                I_WITH_PARTITION = -I$(METIS_PATH)/Lib \
                    -I../../CCOLAMD/Include -I../../CAMD/Include
                CONFIG =
            endif
        endif
    endif
endif
#-------------------------------------------------------------------------------

CLIB = ../Lib/libspqr.a \
        ../../SuiteSparse_config/libsuitesparseconfig.a \
	../../CHOLMOD/Lib/libcholmod.a ../../AMD/Lib/libamd.a \
	../../COLAMD/Lib/libcolamd.a $(LIB_WITH_PARTITION)

# use the BLAS and LAPACK defined by SuiteSparse_config.mk; do not use valgrind 
FLIB = $(LAPACK) $(BLAS)
V = 

# To use Valgrind and the plain BLAS and plain LAPACK (non-optimized):
# FLIB = -lgfortran -llapack_plain -lblas_plain -lg2c
# V = valgrind --quiet

all: library qrdemo qrsimple qrsimplec
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

../Lib/libspqr.a:
	( cd ../Lib ; $(MAKE) )

../../AMD/Lib/libamd.a:
	( cd ../../AMD ; $(MAKE) library )

../../SuiteSparse_config/libsuitesparseconfig.a:
	( cd ../../SuiteSparse_config ; $(MAKE) library )

../../CHOLMOD/Lib/libcholmod.a:
	- ( cd ../../CHOLMOD && $(MAKE) library )

../../COLAMD/Lib/libcolamd.a:
	- ( cd ../../COLAMD && $(MAKE) library )

../../CCOLAMD/Lib/libccolamd.a:
	- ( cd ../../CCOLAMD && $(MAKE) library )

../../CAMD/Lib/libcamd.a:
	- ( cd ../../CAMD && $(MAKE) library )

$(METIS):
	( cd $(METIS_PATH) && $(MAKE) )

library: $(CLIB)

purge: distclean

distclean: clean
	- $(RM) qrdemo qrdemoc qrsimple qrsimplec
	- $(RM) -r *.dSYM

clean:
	- $(RM) $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp Makefile

I = -I../../CHOLMOD/Include -I../../SuiteSparse_config -I../Include

C = $(CXX) $(CF) $(SPQR_CONFIG) $(CONFIG) $(I)

LIBS = $(CLIB) $(LIB) $(FLIB) $(TBB)

qrsimple: qrsimple.cpp library $(INC)
	$(C) qrsimple.cpp -o qrsimple $(LIBS)

qrdemo: qrdemo.cpp library $(INC)
	$(C) qrdemo.cpp -o qrdemo $(LIBS)

# compile the C code with gcc, but link with g++ to use SuiteSparseQR:

qrdemoc.o: qrdemoc.c $(INC)
	$(CC) $(CF) $(SPQR_CONFIG) -ansi $(I) -c $<

qrdemoc: qrdemoc.o library $(INC)
	$(C) -o qrdemoc qrdemoc.o $(LIBS)

qrsimplec.o: qrsimplec.c $(INC)
	$(CC) $(CF) $(SPQR_CONFIG) -ansi $(I) -c $<

qrsimplec: qrsimplec.o library $(INC)
	$(C) -o qrsimplec qrsimplec.o $(LIBS)

