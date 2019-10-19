#===============================================================================
# CHOLMOD/Demo/Makefile
#===============================================================================

# If you compile CHOLMOD with -DNPARTITION, then you do not need METIS or
# CCOLAMD. 

default: all

include ../../UFconfig/UFconfig.mk

#-------------------------------------------------------------------------------
# With METIS, CCOLAMD, CAMD, and the Partition Module:
LIB2 = ../Lib/libcholmod.a ../../AMD/Lib/libamd.a ../../COLAMD/Lib/libcolamd.a \
	../../CCOLAMD/Lib/libccolamd.a ../../CAMD/Lib/libcamd.a \
	$(METIS) $(LAPACK) $(BLAS) $(XERBLA) $(LIB)
# Use this instead, if you compile with -DNPARTITION:
# LIB2 = ../Lib/libcholmod.a ../../AMD/Lib/libamd.a ../../COLAMD/libcolamd.a \
	$(LAPACK) $(BLAS) $(XERBLA) $(LIB)
#-------------------------------------------------------------------------------

C = $(CC) $(CFLAGS) $(CHOLMOD_CONFIG)

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

clean:
	- $(RM) $(CLEAN)

#-------------------------------------------------------------------------------
# See below if you compile with -DNPARTITION
library:
	( cd ../../UFconfig/xerbla ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../CAMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
	( cd ../../CCOLAMD ; $(MAKE) library )
# use this rule instead, if you compile with -DNPARTITION:
# library:
# 	( cd ../../UFconfig/xerbla ; $(MAKE) )
# 	( cd ../Lib ; $(MAKE) )
# 	( cd ../../AMD ; $(MAKE) library )
# 	( cd ../../COLAMD ; $(MAKE) )
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
