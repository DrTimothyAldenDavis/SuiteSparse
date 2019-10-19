This is the UMFPACK Tcov directory.  It runs a large number of tests on
UMFPACK and checks the statement coverage (using gcc and gcov on Linux,
or tcov on Solaris).

You must first do "make purge" in AMD and UMFPACK.  You must also make
sure the "Out" symbolic link is a valid link.  It should point to a large
scratch space, for temporary files.  Finally, type DO.linux or DO.solaris.

Alternatively, just type "make" in this directory, for Linux, or "make sol"
for Solaris.

The last line of each */ut.out file should read
ALL TESTS PASSED largest maxrnorm 1e-07
These lines are summarized at the end of the "DO.linux" test.

If you see "TEST FAILURE" then something went wrong.  "ERROR" messages
in the output files tmp/*.out are OK.  Those are supposed to be there;
the test exercises the error-reporting features of UMFPACK.

DO.all		does all tests

DO 1 di		runs one test (Make.1 and GNUmakefile.di, in this case)

Out/*		subdirectories for each test,
		contents can be destroyed when done.

Make.1		gcc, no optimize Linux no BLAS, test int overflow
Make.2		gcc, no optimize Linux BLAS
Make.3		gcc, no optimize Linux ATLAS C-Blas
Make.4		gcc, no optimize Linux ATLAS Fortran BLAS

Make.5		gcc, no optimize Linux no BLAS, test int overflow
		no reciprocal

Make.1i		icc, optimize, Linux no BLAS, test int overflow
Make.2i		icc, optimize, Linux BLAS
Make.3i		icc, optimize, Linux ATLAS C-Blas
Make.4i		icc, optimize, Linux ATLAS Fortran BLAS

Make.1n		icc, no optimize, Linux no BLAS, test int overflow
Make.2n		icc, no optimize, Linux BLAS
Make.3n		icc, no optimize, Linux ATLAS C-Blas
Make.4n		icc, no optimize, Linux ATLAS Fortran BLAS

Make.1g		gcc, optimize, Linux no BLAS, test int overflow
Make.2g		gcc, optimize, Linux BLAS
Make.3g		gcc, optimize, Linux ATLAS C-Blas
Make.4g		gcc, optimize, Linux ATLAS Fortran BLAS

Make.s1		tcov, Solaris 32 bit, no BLAS
Make.s2		tcov, Solaris 32 bit, Sunperf BLAS
Make.s3		tcov, Solaris 64 bit, no BLAS
Make.s4		tcov, Solaris 64 bit, Sunperf BLAS

Make.s5		tcov, Solaris 32 bit, no BLAS, test int overflow, no recip.

Make.n1		optimize, Solaris 32 bit, no BLAS
Make.n2		optimize, Solaris 32 bit, Sunperf BLAS
Make.n3		optimize, Solaris 64 bit, no BLAS
Make.n4		optimize, Solaris 64 bit, Sunperf BLAS

Makefile.di	Makefile for *di (double, int)
Makefile.dl	Makefile for *dl (double, UF_long)
Makefile.zi	Makefile for *zi (complex, int)
Makefile.zl	Makefile for *zl (complex, UF_long)

TestMat		test matrices for ut.c

../../UMFPACK	UMFPACK original distribution
../../AMD	AMD original distribution
../../UFconfig	configuration directory for all of UFsparse

covall		for summarizing tcov output

ut.c		the test program

