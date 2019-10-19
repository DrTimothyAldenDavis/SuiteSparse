#-------------------------------------------------------------------------------
# Makefile for the LDL library
#-------------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../../SuiteSparse_config -I../Include

C = $(CC) $(CF) $(I)

all: libldl.a

#-------------------------------------------------------------------------------
# the ldl library:
#-------------------------------------------------------------------------------

libldl.a: ../Source/ldl.c ../Include/ldl.h
	$(C) -c ../Source/ldl.c -o ldl.o
	$(C) -DLDL_LONG -c ../Source/ldl.c -o ldll.o
	$(ARCHIVE)  libldl.a ldl.o ldll.o
	- $(RANLIB) libldl.a

distclean: purge

purge: clean
	- $(RM) libldl.a

clean:
	- $(RM) $(CLEAN)

