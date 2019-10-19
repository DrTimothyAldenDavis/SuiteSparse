#------------------------------------------------------------------------------
# Makefile for the CAMD MATLAB mexFunction
#------------------------------------------------------------------------------

default: camd

include ../../UFconfig/UFconfig.mk

CAMD = ../Lib/libcamd.a

I = -I../Include -I../../UFconfig

INC = ../Include/camd.h ../Include/camd_internal.h ../../UFconfig/UFconfig.h

SRC = ../Source/camd_1.c ../Source/camd_2.c ../Source/camd_aat.c \
    ../Source/camd_control.c ../Source/camd_defaults.c ../Source/camd_dump.c \
    ../Source/camd_global.c ../Source/camd_info.c ../Source/camd_order.c \
    ../Source/camd_postorder.c \
    ../Source/camd_preprocess.c ../Source/camd_valid.c

camd: $(SRC) $(INC) camd_mex.c
	$(MEX) -DDLONG $(I) -output camd camd_mex.c $(SRC)

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) camd.mex*
