#------------------------------------------------------------------------------
# Makefile for the AMD MATLAB mexFunction
#------------------------------------------------------------------------------

default: amd2

include ../../UFconfig/UFconfig.mk

AMD = ../Lib/libamd.a

INC = -I../Include -I../../UFconfig

amd2: amd_mex.c
	( cd ../Source ; $(MAKE) )
	$(MEX) $(INC) -output amd2 amd_mex.c $(AMD)

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) amd2.mex* amd2.dll
