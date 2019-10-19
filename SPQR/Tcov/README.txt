SuiteSparseQR exhaustive statement coverage tests.

This test exercises all of SuiteSparseQR and checks its results.  On a 32-bit
platform, all lines are tested.  Some lines of code that handle integer
overflow are not tested (nor are they likely to be testable) on 64-bit
platforms.

A few lines of code in SuiteSparseQR are provably dead code.  This is
intentional; the code has been left for future use.  They are marked with
the comment "// DEAD", and excluded from the test coverage summary.

To run the Tcov tests, first edit the SPQR/Tcov/Makefile to select the
right BLAS and LAPACK libraries (use non-optimized versions for best results).
The just type "make" in this directory.  The output should look like the
make_output.txt file.

Requires Linux; might also work on other Unix platforms.

Files in this directory:

    cov                 csh script that summarizes the statement coverage
    Makefile            for compiling and running the test
    make_output.txt     output of "make"
    matrix1.txt         input file for qrtest, a single matrix
    matrixlist.txt      input file for qrtest, all matrices in ../Matrix
    qrtestc.c           for testing the C interface
    qrtest.cpp          the test program itself.  Rather lengthy.
    README.txt          this file

Timothy A. Davis, http://www.suitesparse.com
