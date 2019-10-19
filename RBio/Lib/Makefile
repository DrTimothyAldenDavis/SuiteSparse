#===============================================================================
# RBio/Lib/Makefile: for compiling the RBio library
#===============================================================================

default: all

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

C = $(CC) $(CF)

all: librbio.a

library: librbio.a

purge: distclean

distclean: clean
	- $(RM) librbio.a

clean:
	- $(RM) $(CLEAN)

#-------------------------------------------------------------------------------
# ../Include/ directory contains all include files:
#-------------------------------------------------------------------------------

INC = ../Include/RBio.h

#-------------------------------------------------------------------------------
# The 7 CHOLMOD library modules (int, double)
#-------------------------------------------------------------------------------

CORE = RBio.o

DI = $(CORE)

LCORE = RBio_long.o

DL = $(LCORE)

#-------------------------------------------------------------------------------

# to compile just the double/int version, use OBJ = $(DI)
OBJ = $(DI)

librbio.a: $(OBJ)
	$(ARCHIVE)  librbio.a $(OBJ)
	- $(RANLIB) librbio.a

$(OBJ): $(INC)

I = -I../Include -I../../SuiteSparse_config

#-------------------------------------------------------------------------------

RBio.o: ../Source/RBio.c
	$(C) -c $(I) $<

RBio_i.o: ../Source/RBio.c
	$(C) -c -DINT $(I) $<

