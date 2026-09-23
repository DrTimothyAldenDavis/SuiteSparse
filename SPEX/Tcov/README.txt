SPEX/Tcov: comprehensive test coverage for SPEX.

Test coverage can be checked with either Linux or Mac.
Type "make" to compile
Requires Linux to run tests with valgrind for memory leak
checking. Type "make vtests" to do so/

The test coverage is in cover.out.  The test output is
printed on stdout, except for cov_test (which prints its output in various
*.out files).

If the test is successful, the last line printed should be
"statements not yet tested: 0", and all printed residuals should be small.
Note that if you are running the test coverage on a macbook, the .out files may
indicate a handful of lines not covered. This is due to a difference between the
gcc and clang compiler. Specifically, there are a few lines containing only
{ or } that the clang compiler classifies as uncovered (for example, the first {
in a switch). These lines are ignored in the "statements not tested" output.

