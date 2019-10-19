# KLU Demo Makefile.  The klu_simple demo is stand-alone.  The other demos
# require CHOLMOD.

default: all

include ../../UFconfig/UFconfig.mk

LIB = ../Lib/libklu.a  ../../BTF/Lib/libbtf.a \
	../../AMD/Lib/libamd.a ../../COLAMD/Lib/libcolamd.a

CHOLMOD = ../../CHOLMOD/Lib/libcholmod.a

I = -I../../UFconfig -I../../AMD/Include -I../../COLAMD/Include \
	-I../../BTF/Include -I../Include -I../../CHOLMOD/Include

all: $(LIB) klu_simple $(CHOLMOD) kludemo kluldemo
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

purge: distclean

distclean: clean
	- $(RM) kludemo kluldemo klu_simple

clean:
	- $(RM) $(CLEAN)

kludemo: kludemo.c Makefile $(LIB)
	$(CC) $(CFLAGS) $(I) kludemo.c -o kludemo $(LIB) $(CHOLMOD) -lm

kluldemo: kludemo.c Makefile $(LIB)
	$(CC) $(CFLAGS) $(I) kluldemo.c -o kluldemo $(LIB) $(CHOLMOD) -lm

klu_simple: klu_simple.c Makefile $(LIB)
	$(CC) $(CFLAGS) $(I) klu_simple.c -o klu_simple $(LIB) -lm
	- ./klu_simple
