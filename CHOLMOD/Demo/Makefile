#===============================================================================
# CHOLMOD/Demo/Makefile
#===============================================================================

# If you compile CHOLMOD with -DNPARTITION, then you do not need METIS or
# CCOLAMD. 

default: all

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

LIB2 = ../Lib/libcholmod.a ../../AMD/Lib/libamd.a ../../COLAMD/Lib/libcolamd.a \
        ../../SuiteSparse_config/libsuitesparseconfig.a \
	$(LIB_WITH_PARTITION) $(LAPACK) $(BLAS) $(XERBLA) $(LIB)

#-------------------------------------------------------------------------------
# With the CUDA BLAS:
ifneq ($(GPU_CONFIG),)
LIB2 += $(GPU_BLAS_PATH)/lib64/libcudart.so $(GPU_BLAS_PATH)/lib64/libcublas.so
endif
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
	( cd ../../SuiteSparse_config/xerbla ; $(MAKE) )
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
ifneq (,$(findstring -DNPARTITION, $(CONFIG)))
else
	( cd ../../CCOLAMD ; $(MAKE) library )
	( cd ../../CAMD ; $(MAKE) library )
	( cd $(METIS_PATH) && $(MAKE) )
endif
#-------------------------------------------------------------------------------

I = -I../Include -I../../SuiteSparse_config

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
