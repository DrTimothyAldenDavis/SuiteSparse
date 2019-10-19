#-------------------------------------------------------------------------------
# COLAMD Makefile
#-------------------------------------------------------------------------------

default: libcolamd.a

include ../../UFconfig/UFconfig.mk

I = -I../Include -I../../UFconfig

INC = ../Include/colamd.h ../../UFconfig/UFconfig.h

SRC = ../Source/colamd.c ../Source/colamd_global.c

# creates libcolamd.a, a C-callable COLAMD library
libcolamd.a:  $(SRC) $(INC)
	$(CC) $(CFLAGS) $(I) -c ../Source/colamd_global.c
	$(CC) $(CFLAGS) $(I) -c ../Source/colamd.c
	$(CC) $(CFLAGS) $(I) -c ../Source/colamd.c -DDLONG -o colamd_l.o
	$(AR) libcolamd.a colamd.o colamd_l.o colamd_global.o

ccode: libcolamd.a

library: libcolamd.a

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) libcolamd.a
