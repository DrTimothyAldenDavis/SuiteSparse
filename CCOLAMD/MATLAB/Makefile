# CCOLAMD Makefile for MATLAB mexFunctions

default: ccolamd2 csymamd2

include ../../UFconfig/UFconfig.mk

I = -I../../UFconfig -I../Include

INC = ../Include/ccolamd.h ../../UFconfig/UFconfig.h

SRC = ../Source/ccolamd.c ../Source/ccolamd_global.c

MX = $(MEX) -DDLONG $(I)

# Compiles the MATLAB-callable routines
mex: ccolamd2 csymamd2

csymamd2: csymamdmex.c $(INC) $(SRC)
	$(MX) -output csymamd csymamdmex.c $(SRC)

ccolamd2: ccolamdmex.c $(INC) $(SRC)
	$(MX) -output ccolamd ccolamdmex.c $(SRC)

# Compiles the extensive test code
test: mex ccolamdtestmex.c csymamdtestmex.c $(INC) $(SRC)
	$(MX) ccolamdtestmex.c $(SRC)
	$(MX) csymamdtestmex.c $(SRC)

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) *.mex* *.dll
