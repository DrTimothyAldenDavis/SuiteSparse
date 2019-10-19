#-------------------------------------------------------------------------------
# AMD Makefile for compiling on Unix systems (for GNU make only)
#-------------------------------------------------------------------------------

default: ../Lib/libamd.a

include ../../UFconfig/UFconfig.mk

C = $(CC) $(CFLAGS) $(CONFIG) -I../Include -I../../UFconfig

#-------------------------------------------------------------------------------
# source files
#-------------------------------------------------------------------------------

AMD = amd_aat amd_1 amd_2 amd_dump amd_postorder amd_post_tree amd_defaults \
	amd_order amd_control amd_info amd_valid amd_preprocess

UFCONFIG = ../../UFconfig/UFconfig.h

INC = ../Include/amd.h ../Include/amd_internal.h $(UFCONFIG)

#-------------------------------------------------------------------------------
# object files for each version
#-------------------------------------------------------------------------------

AMDI = $(addsuffix .o, $(subst amd_,amd_i_,$(AMD)))
AMDL = $(addsuffix .o, $(subst amd_,amd_l_,$(AMD)))

#-------------------------------------------------------------------------------
# compile each int and long routine (with no real/complex version)
#-------------------------------------------------------------------------------

amd_global.o: ../Source/amd_global.c $(INC)
	$(C) -c $< -o $@

amd_i_%.o: ../Source/amd_%.c $(INC)
	$(C) -DDINT -c $< -o $@

amd_l_%.o: ../Source/amd_%.c $(INC)
	$(C) -DDLONG -c $< -o $@

#-------------------------------------------------------------------------------
# Create the libamd.a library (C versions only)
#-------------------------------------------------------------------------------

../Lib/libamd.a: amd_global.o $(AMDI) $(AMDL)
	$(AR) ../Lib/libamd.a $^
	- $(RANLIB) ../Lib/libamd.a

#-------------------------------------------------------------------------------
# compile the Fortran versions and the libamdf77.a library
#-------------------------------------------------------------------------------

fortran: ../Lib/libamdf77.a

AMDF77 = amd.o amdbar.o

amd.o: ../Source/amd.f
	$(F77) $(F77FLAGS) -c ../Source/amd.f -o amd.o

amdbar.o: ../Source/amdbar.f
	$(F77) $(F77FLAGS) -c ../Source/amdbar.f -o amdbar.o

../Lib/libamdf77.a: $(AMDF77)
	$(AR) ../Lib/libamdf77.a $^
	- $(RANLIB) ../Lib/libamdf77.a

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) ../Lib/libamd.a ../Lib/libamdf77.a
