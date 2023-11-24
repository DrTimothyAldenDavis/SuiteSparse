Tcov directory:  Torture test for CHOLMOD, with statement coverage.
http://www.suitesparse.com
--------------------------------------------------------------------------------

This test suite is not required to compile and use CHOLMOD.  It is thus not
ported to all architectures.  Linux is required, and a recent version of
gcc (with the -fprofile-abs-path option).

Requires all CHOLMOD modules: AMD, CAMD, COLAMD, CCOLAMD, and
SuiteSparse_config.  This test acts as a full statement coverage test for all
of these packages.

Type "make" in this directory to compile CHOLMOMD with statement coverage
testing, and to run the tests.  Type ./go to test the GPU.

Every line of CHOLMOD, AMD, CAMD, COLAMD, CCOLAMD, and SuiteSparse_config will
be exercised, and their results checked.  Some matrices will report NaN as
their maximum error (Matrix/c30lo_singular and Matrix/cza for example).  These
test results are expected (see the -s flag on the test input commands).

Note that many, many error messages will appear in the test output itself
(../build/T/*.out), because all of CHOLMOD's error handling is checked as well.
These errors are expected.  Any unexpected error will cause the test to fail.
The last line of each output file should be "All tests passed", and each test
prints "Test OK" or "Test FAIL" on stderr.  If any tests pass, the 'make' will
halt on that test.

This test takes about 20 to 25 minutes.

To remove all but the original source files type "make clean".

