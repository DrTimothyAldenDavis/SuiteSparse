
default: ccolamd_example ccolamd_l_example libccolamd.a

include ../UFconfig/UFconfig.mk

I = -I../UFconfig

ccolamd_example:  ccolamd_example.c libccolamd.a
	$(CC) $(CFLAGS) $(I) -o ccolamd_example ccolamd_example.c libccolamd.a -lm
	- ./ccolamd_example > my_ccolamd_example.out
	- diff ccolamd_example.out my_ccolamd_example.out

ccolamd_l_example:  ccolamd_l_example.c libccolamd.a
	$(CC) $(CFLAGS) $(I) -o ccolamd_l_example ccolamd_l_example.c libccolamd.a -lm
	- ./ccolamd_l_example > my_ccolamd_l_example.out
	- diff ccolamd_l_example.out my_ccolamd_l_example.out

purge: distclean

distclean: clean2
	- $(RM) libccolamd.a

clean2: clean
	- $(RM) *.o *.dll ccolamd_example ccolamd_l_example
	- $(RM) ccolamd.mex* csymamd.mex*
	- $(RM) ccolamdtestmex.mex* csymamdtestmex.mex*
	- $(RM) my_ccolamd_example.out my_ccolamd_l_example.out

# Compiles the MATLAB-callable routines
mex: ccolamdmex.c csymamdmex.c libccolamd.a
	$(MEX) $(I) -output ccolamd ccolamdmex.c libccolamd.a
	$(MEX) $(I) -output csymamd csymamdmex.c libccolamd.a

# Compiles the extensive test code
test: mex ccolamdtestmex.c csymamdtestmex.c libccolamd.a
	$(MEX) $(I) ccolamdtestmex.c libccolamd.a
	$(MEX) $(I) csymamdtestmex.c libccolamd.a

# creates libccolamd.a, a C-callable CCOLAMD library
libccolamd.a:  ccolamd.c ccolamd_global.c ccolamd.h
	$(CC) $(CFLAGS) $(I) -c ccolamd_global.c
	$(CC) $(CFLAGS) $(I) -c ccolamd.c
	$(CC) $(CFLAGS) $(I) -c ccolamd.c -DDLONG -o ccolamd_l.o
	$(AR) libccolamd.a ccolamd.o ccolamd_l.o ccolamd_global.o

ccode: libccolamd.a

library: libccolamd.a

clean:
	- $(RM) $(CLEAN)
