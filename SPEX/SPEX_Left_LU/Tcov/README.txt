SPEX/SPEX/SPEX_Left_LU/Tcov: comprehensive test coverage for SPEX Left LU.

Requires Linux. Type "make" to compile, and then "make run" to run the
tests, or "make vtests" to run the tests with valgrind for memory leakage
checking. 

The test coverage is in cover.out.  The test output is
printed on stdout, except for cov_test (which prints its output in various
*.out files).

If the test is successful, the last line printed should be
"statements not yet tested: 0", and all printed residuals should be small.

