
default: all

ccode: all

include ../../UFconfig/UFconfig.mk

all: kludemo

I = -I../Include -I../../AMD/Include -I../../COLAMD -I../../BTF/Include \
	-I../User -I../../CHOLMOD/Include -I$(METIS_PATH) -I../../CCOLAMD \
	-I../../CAMD -I../../UFconfig

LIBS = ../Lib/libklu.a ../../AMD/Lib/libamd.a ../../COLAMD/libcolamd.a \
	../../BTF/Lib/libbtf.a ../User/libklu_cholmod.a \
	../../CHOLMOD/Lib/libcholmod.a ../../CCOLAMD/libccolamd.a \
	../../CAMD/Lib/libcamd.a $(METIS) $(LIB)

kludemo: library kludemo.c
	$(CC) $(CFLAGS) $(I) kludemo.c dsecnd.c -o kludemo $(LIBS)

purge: clean
	- $(RM) kludemo

distclean: purge

clean:
	- $(RM) $(CLEAN)

library:
	( cd ../Lib ; $(MAKE))
	( cd ../../AMD ; $(MAKE) library)
	( cd ../../CAMD ; $(MAKE) library)
	( cd ../../COLAMD ; $(MAKE) library)
	( cd ../../CCOLAMD ; $(MAKE) library)
	( cd ../../BTF ; $(MAKE))
	( cd ../../CHOLMOD/Lib ; $(MAKE))
	( cd ../User ; $(MAKE))
