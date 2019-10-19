#-------------------------------------------------------------------------------
# CAMD Makefile for compiling on Unix systems (for GNU make only)
#-------------------------------------------------------------------------------

default: ../Lib/libcamd.a

include ../../UFconfig/UFconfig.mk

C = $(CC) $(CFLAGS) $(CONFIG) -I../Include -I../../UFconfig

#-------------------------------------------------------------------------------
# source files
#-------------------------------------------------------------------------------

CAMD = camd_aat camd_1 camd_2 camd_dump camd_postorder camd_defaults \
	camd_order camd_control camd_info camd_valid camd_preprocess

UFCONFIG = ../../UFconfig/UFconfig.h

INC = ../Include/camd.h ../Include/camd_internal.h $(UFCONFIG)

#-------------------------------------------------------------------------------
# object files for each version
#-------------------------------------------------------------------------------

CAMDI = $(addsuffix .o, $(subst camd_,camd_i_,$(CAMD)))
CAMDL = $(addsuffix .o, $(subst camd_,camd_l_,$(CAMD)))

#-------------------------------------------------------------------------------
# compile each int and long routine (with no real/complex version)
#-------------------------------------------------------------------------------

camd_global.o: camd_global.c $(INC)
	$(C) -c $< -o $@

camd_i_%.o: camd_%.c $(INC)
	$(C) -DDINT -c $< -o $@

camd_l_%.o: camd_%.c $(INC)
	$(C) -DDLONG -c $< -o $@

#-------------------------------------------------------------------------------
# Create the libcamd.a library (C versions only)
#-------------------------------------------------------------------------------

../Lib/libcamd.a: camd_global.o $(CAMDI) $(CAMDL)
	$(AR) ../Lib/libcamd.a $^
	- $(RANLIB) ../Lib/libcamd.a

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) ../Lib/libcamd.a
