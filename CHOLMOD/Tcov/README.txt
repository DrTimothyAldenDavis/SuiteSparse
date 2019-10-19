Tcov directory:  Torture test for CHOLMOD, with statement coverage.
http://www.suitesparse.com
--------------------------------------------------------------------------------

This test suite is not required to compile and use CHOLMOD.  It is thus
not ported to all architectures.  Linux is required.

Requires all CHOLMOD modules except the Partition Module, which it can
optionally use (and test).  Also acts as a statement coverage test for
AMD, COLAMD, and CCOLAMD.

Type "make" in this directory to compile CHOLMOMD with statement coverage
testing, and to run the tests.  Type "make go" to rerun the tests.

Note that about 500MB of disk space is required, mostly in the
CHOLMOD_TCOV_TMP directory.

Every line of AMD, CAMD, COLAMD, CCOLAMD, and CHOLMOD will be exercised,
and their results checked.  The line "All tests passed" should be
printed for each test on stderr.  Some matrices will report NaN as their
maximum error; these are the four singular test matrices (Matrix/1_0,
Matrix/3singular, Matrix/c3singluar, and Matrix/z3singular).  These test
results are expected.  Nan's appear in CHOLMOD_TCOV_TMP/galenet_nan.out
and CHOLMOD_TCOV_TMP/l_galenet_nan.out; these are generated intentionally,
to test the code's NaN-handling features.

The source code files are first preprocessed with cc -E, and the resulting
file (z_*.c, zz_*.c, l_*.c, or zl_*.c) is then compiled.  This is to ensure
that all lines within macros and included *.c files are tested (take a look at
z_updown.c and z_solve.c if you'd like to see how loop-unrolling and
real/complex templates are done in CHOLMOD, and compare those files with their
source files in ../Modify and ../Cholesky).

Note that many, many error messages will appear in the test output itself
(CHOLMOD_TCOV_TMP/*.out), because all of CHOLMOD's error handling is
checked as well.  These errors are expected.  Any unexpected error will cause
the test to fail.  The last line of each output file should be "All tests
successful".

To remove all but the original source files and output files from
this directory, type "make clean".  To remove all but the
files in the original distribution, type "make distclean".

On the Mac (OSX 10.6.1, Snow Leopard), you may see errors like this:

    cl(32662) malloc: *** mmap(size=121600000000000) failed (error code=12)
    *** error: can't allocate region
    *** set a breakpoint in malloc_error_break to debug

    That is not an error.  The test code is rigorously testing the CHOLMOD
    memory management wrappers, by trying to allocate a huge amount of space
    with the expectation that it must fail.  This to ensure the memory
    management routines properly handle that case.  For some reason unknown to
    me, the Mac "malloc" function feels the need to print an error on stdout
    when attempting to malloc something too big.  It should simply and quietly
    return a NULL instead, as Linux does.  Thus, ignore these errors.

