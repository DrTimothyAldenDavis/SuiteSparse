#-----------------------------------------------------------------------------
# compile the AMD demo (for both GNU make or original make)
#-----------------------------------------------------------------------------

default: amd_simple amd_demo amd_demo2 amd_l_demo

include ../../UFconfig/UFconfig.mk

C = $(CC) $(CFLAGS) -I../Include -I../../UFconfig

INC = ../Include/amd.h ../../UFconfig/UFconfig.h

library:
	( cd ../Lib ; $(MAKE) )

f77lib:
	( cd ../Lib ; $(MAKE) fortran )

#------------------------------------------------------------------------------
# Create the demo program, run it, and compare the output
#------------------------------------------------------------------------------

dist:

amd_demo: amd_demo.c library $(INC)
	$(C) -o amd_demo amd_demo.c ../Lib/libamd.a $(LIB)
	./amd_demo > my_amd_demo.out
	- diff amd_demo.out my_amd_demo.out

amd_l_demo: amd_l_demo.c library $(INC)
	$(C) -o amd_l_demo amd_l_demo.c ../Lib/libamd.a $(LIB)
	./amd_l_demo > my_amd_l_demo.out
	- diff amd_l_demo.out my_amd_l_demo.out

amd_demo2: amd_demo2.c library $(INC)
	$(C) -o amd_demo2 amd_demo2.c ../Lib/libamd.a $(LIB)
	./amd_demo2 > my_amd_demo2.out
	- diff amd_demo2.out my_amd_demo2.out

amd_simple: amd_simple.c library $(INC)
	$(C) -o amd_simple amd_simple.c ../Lib/libamd.a $(LIB)
	./amd_simple > my_amd_simple.out
	- diff amd_simple.out my_amd_simple.out

#------------------------------------------------------------------------------
# compile the Fortran demo
#------------------------------------------------------------------------------

fortran: amd_f77demo amd_f77simple

cross: amd_f77cross

amd_f77demo: amd_f77demo.f f77lib
	$(F77) $(F77FLAGS) -o amd_f77demo amd_f77demo.f ../Lib/libamdf77.a \
	    $(F77LIB)
	./amd_f77demo > my_amd_f77demo.out
	- diff amd_f77demo.out my_amd_f77demo.out

amd_f77simple: amd_f77simple.f f77lib
	$(F77) $(F77FLAGS) -o amd_f77simple amd_f77simple.f \
	    ../Lib/libamdf77.a $(F77LIB)
	./amd_f77simple > my_amd_f77simple.out
	- diff amd_f77simple.out my_amd_f77simple.out

amd_f77wrapper.o: amd_f77wrapper.c
	$(C) -DDINT -c amd_f77wrapper.c

amd_f77cross:  amd_f77cross.f amd_f77wrapper.o ../Lib/libamd.a
	$(F77) $(F77FLAGS) -o amd_f77cross amd_f77cross.f amd_f77wrapper.o \
	    ../Lib/libamd.a $(F77LIB)
	./amd_f77cross > my_amd_f77cross.out
	- diff amd_f77cross.out my_amd_f77cross.out

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) amd_demo my_amd_demo.out
	- $(RM) amd_l_demo my_amd_l_demo.out
	- $(RM) amd_demo2 my_amd_demo2.out
	- $(RM) amd_simple my_amd_simple.out
	- $(RM) amd_f77demo my_amd_f77demo.out
	- $(RM) amd_f77simple my_amd_f77simple.out
	- $(RM) amd_f77cross my_amd_f77cross.out
