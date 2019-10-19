# KLU Demo Makefile.  The klu_simple demo is stand-alone.  The other demos
# require CHOLMOD.

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

CLIB = ../Lib/libklu.a  ../../BTF/Lib/libbtf.a \
	../../AMD/Lib/libamd.a ../../COLAMD/Lib/libcolamd.a \
        ../../SuiteSparse_config/libsuitesparseconfig.a

CHOLMOD_LIB = ../../CHOLMOD/Lib/libcholmod.a
CHOLMOD = $(CHOLMOD_LIB) $(LAPACK) $(BLAS) $(XERBLA)

ifneq ($(GPU_CONFIG),)
CHOLMOD += $(CUDART_LIB) $(CUBLAS_LIB) 
endif

I = -I../../SuiteSparse_config -I../../AMD/Include -I../../COLAMD/Include \
	-I../../BTF/Include -I../Include -I../../CHOLMOD/Include

all: $(CLIB) klu_simple $(CHOLMOD_LIB) kludemo kluldemo
	- ./klu_simple
	- ./kludemo < ../Matrix/1c.mtx
	- ./kludemo < ../Matrix/arrowc.mtx
	- ./kludemo < ../Matrix/arrow.mtx
	- ./kludemo < ../Matrix/impcol_a.mtx
	- ./kludemo < ../Matrix/w156.mtx
	- ./kludemo < ../Matrix/ctina.mtx
	- ./kluldemo < ../Matrix/1c.mtx
	- ./kluldemo < ../Matrix/arrowc.mtx
	- ./kluldemo < ../Matrix/arrow.mtx
	- ./kluldemo < ../Matrix/impcol_a.mtx
	- ./kluldemo < ../Matrix/w156.mtx
	- ./kluldemo < ../Matrix/ctina.mtx

../Lib/libklu.a:
	( cd ../Lib ; $(MAKE) )

../../BTF/Lib/libbtf.a:
	( cd ../../BTF ; $(MAKE) library )

../../AMD/Lib/libamd.a:
	( cd ../../AMD ; $(MAKE) library )

../../COLAMD/Lib/libcolamd.a:
	( cd ../../COLAMD ; $(MAKE) library )

../../CHOLMOD/Lib/libcholmod.a:
	( cd ../../CHOLMOD ; $(MAKE) library )
#	( cd ../../CAMD ; $(MAKE) )
#	( cd ../../CCOLAMD ; $(MAKE) )
#	( cd ../../metis-4.0 ; $(MAKE) )

../../SuiteSparse_config/libsuitesparseconfig.a:
	( cd ../../SuiteSparse_config ; $(MAKE) )

purge: distclean

distclean: clean
	- $(RM) kludemo kluldemo klu_simple
	- $(RM) -r *.dSYM

clean:
	- $(RM) $(CLEAN)

kludemo: kludemo.c Makefile $(CLIB)
	$(CC) $(CF) $(I) kludemo.c -o kludemo $(CLIB) $(CHOLMOD) $(LIB)

kluldemo: kludemo.c Makefile $(CLIB)
	$(CC) $(CF) $(I) kluldemo.c -o kluldemo $(CLIB) $(CHOLMOD) $(LIB)

klu_simple: klu_simple.c Makefile $(CLIB)
	$(CC) $(CF) $(I) klu_simple.c -o klu_simple $(CLIB) $(LIB)
	- ./klu_simple
