# COLAMD Makefile for MATLAB mexFunctions

default: colamd2 symamd2

include ../../UFconfig/UFconfig.mk

I = -I../../UFconfig -I../Include

INC = ../Include/colamd.h ../../UFconfig/UFconfig.h

SRC = ../Source/colamd.c ../Source/colamd_global.c

MX = $(MEX) -DDLONG $(I)

# Compiles the MATLAB-callable routines
mex: colamd2 symamd2

symamd2: symamdmex.c $(INC) $(SRC)
	$(MX) -output symamd2mex symamdmex.c $(SRC)

colamd2: colamdmex.c $(INC) $(SRC)
	$(MX) -output colamd2mex colamdmex.c $(SRC)

# Compiles the extensive test code
test: mex colamdtestmex.c symamdtestmex.c $(INC) $(SRC)
	$(MX) colamdtestmex.c $(SRC)
	$(MX) symamdtestmex.c $(SRC)

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) *.mex* *.dll
