#------------------------------------------------------------------------------
# Makefile for the AMD MATLAB mexFunction
#------------------------------------------------------------------------------

default: amd

include ../../UFconfig/UFconfig.mk

AMD = ../Lib/libamd.a

INC = -I../Include -I../../UFconfig

amd: amd_mex.c
	( cd ../Source ; $(MAKE) )
	$(MEX) $(INC) -output amd amd_mex.c $(AMD)

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) amd.mex* amd.dll
