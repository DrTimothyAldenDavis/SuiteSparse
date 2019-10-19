
default: libcolamd.a colamd_example colamd_l_example

include ../UFconfig/UFconfig.mk

I = -I../UFconfig

colamd_example: colamd_example.c libcolamd.a
	$(CC) $(CFLAGS) $(I) -o colamd_example colamd_example.c libcolamd.a -lm
	- ./colamd_example > my_colamd_example.out
	- diff colamd_example.out my_colamd_example.out

colamd_l_example: colamd_l_example.c libcolamd.a
	$(CC) $(CFLAGS) $(I) -o colamd_l_example colamd_l_example.c libcolamd.a -lm
	- ./colamd_l_example > my_colamd_l_example.out
	- diff colamd_example.out my_colamd_example.out

purge: distclean

distclean: clean2
	- $(RM) libcolamd.a

clean2: clean
	- $(RM) *.o *.dll colamd_example colamd_l_example
	- $(RM) colamd2mex.mex* symamd2mex.mex*
	- $(RM) colamdtestmex.mex* symamdtestmex.mex*
	- $(RM) my_colamd_example.out my_colamd_l_example.out

# Compiles the MATLAB-callable routines
mex: colamdmex.c symamdmex.c libcolamd.a
	$(MEX) -output colamd2mex $(I) colamdmex.c libcolamd.a
	$(MEX) -output symamd2mex $(I) symamdmex.c libcolamd.a

# Compiles the extensive test code
test: mex colamdtestmex.c symamdtestmex.c libcolamd.a
	$(MEX) $(I) colamdtestmex.c libcolamd.a
	$(MEX) $(I) symamdtestmex.c libcolamd.a

# creates libcolamd.a, a C-callable COLAMD library
libcolamd.a:  colamd.c colamd_global.c colamd.h
	$(CC) $(CFLAGS) $(I) -c colamd_global.c
	$(CC) $(CFLAGS) $(I) -c colamd.c
	$(CC) $(CFLAGS) $(I) -c colamd.c -DDLONG -o colamd_l.o
	$(AR) libcolamd.a colamd.o colamd_l.o colamd_global.o

ccode: libcolamd.a

library: libcolamd.a

clean:
	- $(RM) $(CLEAN)
