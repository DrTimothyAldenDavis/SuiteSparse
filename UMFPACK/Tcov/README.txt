This is the UMFPACK Tcov directory.  It runs a large number of tests on
UMFPACK and checks the statement coverage (using gcc and gcov on Linux
or the Mac, or tcov on Solaris).

Timothy A. Davis, http://www.suitesparse.com

METIS is required for this test.

You must first do "make purge" in AMD and UMFPACK.  Then type "make" in
this directory.

You terminal output will look something like this:

################################################################################
Tcov test: 1 di
################################################################################
make[2]: [run] Error 1 (ignored)
make[2]: [run] Error 1 (ignored)
make[2]: [run] Error 1 (ignored)
make[2]: [run] Error 1 (ignored)
ALL TESTS PASSED: rnorm 2.22e-10 (1.79e-07 shl0, 9.16e-05 arc130 8.81e-08 omega2) cputime 45.7908
58.048u 26.549s 1:59.54 70.7%   0+0k 3+340io 0pf+0w
################################################################################
Tcov test: 1 dl
################################################################################
...

The ignored errors are OK (these are from diff's that do not need to be
identical, since the output files include timing information, sizeof(void *)
which can differ on different platforms, and so on).  You may also see warnings
that *dump.o files have no symbols.  This is normal.  These files contain
debugging functions only, and debugging is disabled by default.

The last line of each */ut.out file should read something like
ALL TESTS PASSED: rnorm 1e-07 ( ... )
These lines are summarized at the end of the "DO.all" test.

If you see "TEST FAILURE" then something went wrong.  "ERROR" messages
in the output files tmp/*.out are OK.  Those are supposed to be there;
the test exercises the error-reporting features of UMFPACK.

Files and directories:

../../UMFPACK	UMFPACK original distribution
../../AMD	AMD original distribution
../../SuiteSparse_config	configuration directory for all of SuiteSparse

acov.di         AMD test coverage scripts
acov.dl
acov.zi
acov.zl

AMD_Demo_Makefile   replacement for AMD/Demo/Makefile, for testing

badnum2.umf     intentionally corrupted files to test umfpack_load_* functions
badnum.umf
badsym2.umf
badsym.umf

cov		for summarizing tcov output
covall
cover.awk

debug.amd
debug.umf

Demo_Makefile   replacement for UMFPACK/Demo/Makefile, for testing

DO              runs one test
DO.all          run all tests

GNUmakefile.di  UMFPACK/Lib/Makefile replacements, for testing
GNUmakefile.dl
GNUmakefile.zi
GNUmakefile.zl

Make.1		no optimize, no BLAS, test int overflow
Make.2		no optimize, BLAS
Make.3		optimize, no BLAS, test int overflow
Make.4		optimize, BLAS
Make.5		no optimize, no BLAS, test int overflow, no reciprocal
Make.6          no optimize, BLAS, no timers
Make.7          no optimize, test for int overflow, no divide-by-zero
Make.8          no optimize, no test for int overflow, no divide-by-zero

Makefile        top-level Makefile for Tcov tests

Out/*		subdirectories for each test,
		contents can be destroyed when done.

README.txt      this file

TestMat		test matrices for ut.c

Top_Makefile    replacement for AMD/Makefile and UMFPACK/Makefile, for testing

ucov.di         UMFPACK test coverage scripts
ucov.dl
ucov.zi
ucov.zl

ut.c		the test program itself

