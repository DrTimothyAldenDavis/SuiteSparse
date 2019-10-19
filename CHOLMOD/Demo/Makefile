#===============================================================================
# CHOLMOD/Demo/Makefile
#===============================================================================

# If you compile CHOLMOD with -DNPARTITION, then you do not need METIS or
# CCOLAMD. 

default: all

include ../../UFconfig/UFconfig.mk

#-------------------------------------------------------------------------------
# the optional Partition module requires METIS, CAMD, and CCOLAMD
LIB_WITH_PARTITION =
CONFIG =
ifeq (,$(findstring -DNPARTITION, $(CHOLMOD_CONFIG)))
    # check if METIS is available
    ifeq (../../metis-4.0, $(wildcard ../../metis-4.0))
        LIB_WITH_PARTITION = $(METIS) ../../CCOLAMD/Lib/libccolamd.a \
            ../../CAMD/Lib/libcamd.a
    else
        # METIS is not available, but CHOLMOD_CONFIG does not specify
        # -DNPARTITION.  Add it here so we can still compile CHOLMOD.
        CONFIG = -DNPARTITION
    endif
endif
#-------------------------------------------------------------------------------

LIB2 = ../Lib/libcholmod.a ../../AMD/Lib/libamd.a ../../COLAMD/Lib/libcolamd.a \
	$(LIB_WITH_PARTITION) $(LAPACK) $(BLAS) $(XERBLA) $(LIB)

#-------------------------------------------------------------------------------

C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(CONFIG)

code: library cholmod_demo cholmod_l_demo cholmod_simple

fortran: readhb readhb2 reade 

all: code
	./cholmod_demo < Matrix/bcsstk01.tri
	./cholmod_l_demo < Matrix/bcsstk01.tri
	./cholmod_demo < Matrix/lp_afiro.tri
	./cholmod_l_demo < Matrix/lp_afiro.tri
	./cholmod_demo < Matrix/can___24.mtx
	./cholmod_l_demo < Matrix/can___24.mtx
	./cholmod_demo < Matrix/c.tri
	./cholmod_l_demo < Matrix/c.tri
	./cholmod_simple < Matrix/c.tri
	./cholmod_simple < Matrix/can___24.mtx
	./cholmod_simple < Matrix/bcsstk01.tri

distclean: purge

purge: clean
	- $(RM) cholmod_demo cholmod_l_demo readhb readhb2 reade
	- $(RM) cholmod_simple
	- $(RM) -r *.dSYM

clean:
	- $(RM) $(CLEAN)

#-------------------------------------------------------------------------------
# See below if you compile with -DNPARTITION
library:
	( cd ../../UFconfig/xerbla ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
ifneq (,$(findstring -DNPARTITION, $(CHOLMOD_CONFIG)))
else
	( cd ../../CCOLAMD ; $(MAKE) library )
	( cd ../../CAMD ; $(MAKE) library )
endif
#-------------------------------------------------------------------------------

I = -I../Include -I../../UFconfig

cholmod_demo: library cholmod_demo.c cholmod_demo.h
	$(C) -o cholmod_demo $(I) cholmod_demo.c $(LIB2)

cholmod_simple: library cholmod_simple.c
	$(C) -o cholmod_simple $(I) cholmod_simple.c $(LIB2)

cholmod_l_demo: library cholmod_l_demo.c cholmod_demo.h
	$(C) -o cholmod_l_demo $(I) cholmod_l_demo.c $(LIB2)

readhb: readhb.f
	$(F77) $(FFLAGS) -o readhb readhb.f

readhb2: readhb2.f
	$(F77) $(FFLAGS) -O -o readhb2 readhb2.f

reade: reade.f
	$(F77) $(FFLAGS) -O -o reade reade.f
