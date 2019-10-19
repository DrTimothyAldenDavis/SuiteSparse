#-------------------------------------------------------------------------------
# AMD Makefile for compiling on Unix systems (for original Make ONLY)
#-------------------------------------------------------------------------------

# This is a very ugly Makefile, and is only provided for those who do not
# have GNU make.  Note that it is not used if you have GNU make.  It ignores
# dependency checking and just compiles everything.

default: everything

include ../../UFconfig/UFconfig.mk

C = $(CC) $(CFLAGS) $(CONFIG) -I../Include -I../../UFconfig

everything:
	$(C) -DDINT -c ../Source/amd_aat.c -o amd_i_aat.o
	$(C) -DDINT -c ../Source/amd_1.c -o amd_i_1.o
	$(C) -DDINT -c ../Source/amd_2.c -o amd_i_2.o
	$(C) -DDINT -c ../Source/amd_dump.c -o amd_i_dump.o
	$(C) -DDINT -c ../Source/amd_postorder.c -o amd_i_postorder.o
	$(C) -DDINT -c ../Source/amd_post_tree.c -o amd_i_post_tree.o
	$(C) -DDINT -c ../Source/amd_defaults.c -o amd_i_defaults.o
	$(C) -DDINT -c ../Source/amd_order.c -o amd_i_order.o
	$(C) -DDINT -c ../Source/amd_control.c -o amd_i_control.o
	$(C) -DDINT -c ../Source/amd_info.c -o amd_i_info.o
	$(C) -DDINT -c ../Source/amd_valid.c -o amd_i_valid.o
	$(C) -DDINT -c ../Source/amd_preprocess.c -o amd_i_preprocess.o
	$(C) -DDLONG -c ../Source/amd_aat.c -o amd_l_aat.o
	$(C) -DDLONG -c ../Source/amd_1.c -o ../Source/amd_l_1.o
	$(C) -DDLONG -c ../Source/amd_2.c -o amd_l_2.o
	$(C) -DDLONG -c ../Source/amd_dump.c -o amd_l_dump.o
	$(C) -DDLONG -c ../Source/amd_postorder.c -o amd_l_postorder.o
	$(C) -DDLONG -c ../Source/amd_post_tree.c -o amd_l_post_tree.o
	$(C) -DDLONG -c ../Source/amd_defaults.c -o amd_l_defaults.o
	$(C) -DDLONG -c ../Source/amd_order.c -o amd_l_order.o
	$(C) -DDLONG -c ../Source/amd_control.c -o amd_l_control.o
	$(C) -DDLONG -c ../Source/amd_info.c -o amd_l_info.o
	$(C) -DDLONG -c ../Source/amd_valid.c -o amd_l_valid.o
	$(C) -DDLONG -c ../Source/amd_preprocess.c -o amd_l_preprocess.o
	$(C) -c ../Source/amd_global.c
	$(AR) ../Lib/libamd.a amd_i_aat.o amd_i_1.o amd_i_2.o amd_i_dump.o \
	    amd_i_postorder.o amd_i_post_tree.o amd_i_defaults.o amd_i_order.o \
	    amd_i_control.o amd_i_info.o amd_i_valid.o amd_l_aat.o amd_l_1.o \
	    amd_l_2.o amd_l_dump.o amd_l_postorder.o amd_l_post_tree.o \
	    amd_l_defaults.o amd_l_order.o amd_l_control.o amd_l_info.o \
	    amd_l_valid.o amd_i_preprocess.o amd_l_preprocess.o amd_global.o
	- $(RANLIB) ../Lib/libamd.a

#-------------------------------------------------------------------------------
# compile the Fortran versions and the libamdf77.a library
#-------------------------------------------------------------------------------

fortran:
	$(F77) $(F77FLAGS) -c ../Source/amd.f -o amd.o
	$(F77) $(F77FLAGS) -c ../Source/amdbar.f -o amdbar.o
	$(AR) ../Lib/libamdf77.a amd.o amdbar.o
	- $(RANLIB) ../Lib/libamdf77.a

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) ../Lib/libamd.a ../Lib/libamdf77.a
