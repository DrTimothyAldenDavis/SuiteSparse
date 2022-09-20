This is the UMFPACK Tcov directory.  It runs a large number of tests on
UMFPACK and checks the statement coverage (using gcc and gcov on Linux).

Timothy A. Davis, http://www.suitesparse.com

To run the tests, type "make" in this directory.  The METIS, CAMD, CCOLAMD,
COLAMD, and CHOLMOD libraries will be compiled with TCOV=yes (using gcc)
and placed in ../../lib.  This will delete any prior compiled libraries placed
there.  Compiled versions of UMFPACK and its output are placed in
$(TCOV_TMP)/UMFPACK_TCOV_TMP, where $(TCOV_TMP) defaults to /tmp
(it can be changed via 'make TCOV_TMP=/home/myself/mytempdirectory' for
example).

To remove the compiled copies (in $(TCOV_TMP)/UMFPACK_TCOV_TMP) do 'make clean'.
This will keep the statement coverage summary, ./cover.out.

To remove all but the distributed files, do 'make purge' or 'make distclean'.

You terminal output will look something like this:

################################################################################
Tcov test: 1 di
################################################################################
ALL TESTS PASSED: rnorm 2.22e-10 (1.79e-07 shl0, 9.16e-05 arc130 8.81e-08 omega2)
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
in the output files $(TCOV_TMP)/*.out are OK.  Those are supposed to be there;
the test exercises the error-reporting features of UMFPACK.

Files and directories:

acov.di         AMD test coverage scripts
acov.dl
acov.zi
acov.zl

badnum2.umf     intentionally corrupted files to test umfpack_load_* functions
badnum.umf
badsym2.umf
badsym.umf

cov		for summarizing tcov output
covall
cover.awk

debug.amd       sets the debug levels in AMD and UMFPACK
debug.umf

DO              runs one test
DO.all          run all tests

Makefile.di     UMFPACK/Lib/Makefile replacements, for testing
Makefile.dl
Makefile.zi
Makefile.zl

Make.1		no optimize, no BLAS, test int overflow
Make.2		no optimize, BLAS
Make.3		optimize, no BLAS, test int overflow
Make.4		optimize, BLAS
Make.5		no optimize, no BLAS, test int overflow, no reciprocal
Make.6          no optimize, BLAS, no timers
Make.7          no optimize, test for int overflow, no divide-by-zero
Make.8          no optimize, no test for int overflow, no divide-by-zero

Makefile        top-level Makefile for Tcov tests

$(TCOV_TMP)/UMFPACK_TCOV_TMP/*	    subdirectories for each test,
		contents can be destroyed when done.

README.txt      this file

TestMat		test matrices for ut.c:

                adlittle
                arc130
                cage3
                d_dyn
                galenet
                matrix1
                matrix10
                matrix11
                matrix12
                matrix13
                matrix14
                matrix15
                matrix16
                matrix17
                matrix18
                matrix19
                matrix2
                matrix20
                matrix21
                matrix22
                matrix23
                matrix24
                matrix25
                matrix26
                matrix27
                matrix28
                matrix29
                matrix3
                matrix30
                matrix4
                matrix5
                matrix6
                matrix7
                matrix8
                nug07
                S_d2q06c
                shl0

Top_Makefile    replacement for AMD/Makefile and UMFPACK/Makefile, for testing

ucov.di         UMFPACK test coverage scripts
ucov.dl
ucov.zi
ucov.zl

ut.c		the test program itself

