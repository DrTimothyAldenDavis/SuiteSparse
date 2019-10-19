default: all

include ../../UFconfig/UFconfig.mk

all: libklu_cholmod.a

I = -I../../CHOLMOD/Include -I../../UFconfig -I../Include -I../../AMD/Include \
    -I../../BTF/Include -I../../COLAMD

libklu_cholmod.a:  library klu_cholmod.c klu_cholmod.h
	$(CC) $(CF) $(I) -c klu_cholmod.c
	$(ARCHIVE)  libklu_cholmod.a klu_cholmod.o
	- $(RANLIB) libklu_cholmod.a

distclean: purge

purge: clean
	- $(RM) *.o *.a

clean:
	- $(RM) $(CLEAN)
	- $(RM) -r *.dSYM

library:
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../COLAMD ; $(MAKE) library )
	( cd ../../CHOLMOD/Lib ; $(MAKE) )
