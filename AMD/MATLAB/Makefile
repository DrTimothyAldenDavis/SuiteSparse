#------------------------------------------------------------------------------
# Makefile for the AMD MATLAB mexFunction
#------------------------------------------------------------------------------

default: amd2

include ../../UFconfig/UFconfig.mk

AMD = ../Lib/libamd.a

I = -I../Include -I../../UFconfig

INC = ../Include/amd.h ../Include/amd_internal.h ../../UFconfig/UFconfig.h

SRC = ../Source/amd_1.c ../Source/amd_2.c ../Source/amd_aat.c \
    ../Source/amd_control.c ../Source/amd_defaults.c ../Source/amd_dump.c \
    ../Source/amd_global.c ../Source/amd_info.c ../Source/amd_order.c \
    ../Source/amd_postorder.c ../Source/amd_post_tree.c \
    ../Source/amd_preprocess.c ../Source/amd_valid.c

amd2: $(SRC) $(INC) amd_mex.c
	$(MEX) -DDLONG $(I) -output amd2 amd_mex.c $(SRC)

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) amd2.mex*
