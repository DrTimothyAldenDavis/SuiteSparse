default: all

include ../../UFconfig/UFconfig.mk

I = -I../Include -I../../AMD/Include -I../../COLAMD -I../../CCOLAMD \
	-I../../CHOLMOD/Include -I../User -I../../BTF/Include \
	-I../../UFconfig -I../../CAMD/Include

LIB = ../Lib/libklu.a \
	../User/libklu_cholmod.a ../../BTF/Lib/libbtf.a \
	../../CHOLMOD/Lib/libcholmod.a $(METIS_PATH)/libmetis.a \
	../../AMD/Lib/libamd.a ../../COLAMD/libcolamd.a  \
	../../CCOLAMD/libccolamd.a ../../CAMD/Lib/libcamd.a

all: klus klu

klus: library klus.c
	$(MEX) klus.c $(I) $(LIB)

klu: library klu_mex.c
	$(MEX) -o klu klu_mex.c $(I) $(LIB)

distclean: purge

purge: clean
	- $(RM) *.a *.o *.mex*

clean:
	- $(RM) $(CLEAN)

library:
	( cd ../../AMD ; $(MAKE) )
	( cd ../../BTF ; $(MAKE) )
	( cd ../../CAMD ; $(MAKE) )
	( cd ../../COLAMD ; $(MAKE) )
	( cd ../../CCOLAMD ; $(MAKE) )
	( cd ../../CHOLMOD/Lib ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
	( cd ../User ; $(MAKE) )
