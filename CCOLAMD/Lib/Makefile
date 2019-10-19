#-------------------------------------------------------------------------------
# CCOLAMD Makefile
#-------------------------------------------------------------------------------

default: libccolamd.a

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../Include -I../../SuiteSparse_config

INC = ../Include/ccolamd.h ../../SuiteSparse_config/SuiteSparse_config.h

SRC = ../Source/ccolamd.c ../Source/ccolamd_global.c

# creates libccolamd.a, a C-callable COLAMD library
libccolamd.a:  $(SRC) $(INC)
	$(CC) $(CF) $(I) -c ../Source/ccolamd_global.c
	$(CC) $(CF) $(I) -c ../Source/ccolamd.c
	$(CC) $(CF) $(I) -c ../Source/ccolamd.c -DDLONG -o ccolamd_l.o
	$(ARCHIVE)  libccolamd.a ccolamd.o ccolamd_l.o ccolamd_global.o
	- $(RANLIB) libccolamd.a 

ccode: libccolamd.a

library: libccolamd.a

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) libccolamd.a
