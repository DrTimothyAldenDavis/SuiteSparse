#-------------------------------------------------------------------------------
# Makefile for the LDL mexFunction
#-------------------------------------------------------------------------------

default: mex

include ../../UFconfig/UFconfig.mk

I = -I../../UFconfig -I../Include

MX = $(MEX) $(I) -DLDL_LONG

mex: ldlmex.c ../Source/ldl.c ../Include/ldl.h ldlsymbolmex.c
	- $(MX) -output ldlsparse ../Source/ldl.c ldlmex.c
	- $(MX) -output ldlsymbol ../Source/ldl.c ldlsymbolmex.c

#-------------------------------------------------------------------------------
# clean-up:
#-------------------------------------------------------------------------------

distclean: purge

purge: clean
	- $(RM) *.mex*

clean:
	- $(RM) $(CLEAN)
