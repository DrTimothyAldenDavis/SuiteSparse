#------------------------------------------------------------------------------
# Makefile for the CAMD MATLAB mexFunction
#------------------------------------------------------------------------------

default: camd

include ../../UFconfig/UFconfig.mk

CAMD = ../Lib/libcamd.a

INC = -I../Include -I../../UFconfig

camd: camd_mex.c
	( cd ../Source ; $(MAKE) )
	$(MEX) $(INC) -output camd camd_mex.c $(CAMD)

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) camd.mex* camd.dll
