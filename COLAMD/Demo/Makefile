#-----------------------------------------------------------------------------
# compile the COLAMD demo
#-----------------------------------------------------------------------------

default: colamd_example colamd_l_example

include ../../UFconfig/UFconfig.mk

I = -I../Include -I../../UFconfig

C = $(CC) $(CFLAGS) $(I)

library:
	( cd ../Lib ; $(MAKE) )

#------------------------------------------------------------------------------
# Create the demo program, run it, and compare the output
#------------------------------------------------------------------------------

dist:

colamd_example: colamd_example.c library
	$(C) -o colamd_example colamd_example.c ../Lib/libcolamd.a -lm
	- ./colamd_example > my_colamd_example.out
	- diff colamd_example.out my_colamd_example.out

colamd_l_example: colamd_l_example.c library
	$(C) -o colamd_l_example colamd_l_example.c ../Lib/libcolamd.a -lm
	- ./colamd_l_example > my_colamd_l_example.out
	- diff colamd_example.out my_colamd_example.out

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) colamd_example colamd_l_example
	- $(RM) my_colamd_example.out my_colamd_l_example.out
