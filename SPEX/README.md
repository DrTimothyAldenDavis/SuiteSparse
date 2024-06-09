SPEX is a software package for SParse EXact algebra

Files and folders in this distribution:

    README.md       this file

    build           Contains the SPEX C library as well
                    as the .so files

    DOC             User guide for the SPEX software package

    MATLAB          MATLAB interface for the SPEX software package

    SPEX_Backslash  Exactly solve sparse linear systems with
                    default settings. This is the easiest
                    starting point for the SPEX software package.
                    SPEX_Backslash will automatically determine the
                    appropriate factorization algorithm for use in
                    solving your problem A x = b (Developmental)

    SPEX_Cholesky   Sparse integer-preserving SPEX_Cholesky
                    factorization for exactly solving SPD
                    linear systems (Developmental)

    SPEX_LU         Sparse left-looking integer-preserving
                    LU factorization for exactly solve
                    sparse linear systems. (Release)

    SPEX_QR         Sparse integer-preserving QR factorization
                    (Developmental)

    SPEX_Update     Sparse column replacement and rank 1 updates
                    for the SPEX factorizations

    SPEX_UTIL       Utility functions for all SPEX components

    Makefile        compiles SPEX and its dependencies

Dependencies:

    AMD                 approximate minimum degree ordering

    COLAMD              column approximate minimum degree ordering

    SuiteSparse_config  configuration for all of SuiteSparse

    GNU GMP             GNU Multiple Precision Arithmetic Library
                        for big integer operations.  v6.1.2 or later
                        is required.

    GNU MPFR            GNU Multiple Precision Floating-Point Reliable
                        Library for arbitrary precision floating point
                        operations. v4.0.2 or later is required.

Compilation options:

* `SPEX_USE_PYTHON`:

  If `ON`, build Python interface for SPEX.
  If `OFF`: do not build the SPEX Python interface.
  Default: `SUITESPARSE_USE_PYTHON` (defaults to ON).

* `SPEX_USE_OPENMP`:

  If `ON`, OpenMP is used in SPEX if it is available.
  Default: `SUITESPARSE_USE_OPENMP` (defaults to ON).

To compile SPEX and its dependencies, just type "make" in this folder.
This will also run a few short demos
To install the package system-wide, copy the `lib/*` to /usr/local/lib,
and copy `include/*` to /usr/local/include.

Authors (alphabetical order):

    Jinhao Chen
    Timothy A. Davis
    Christopher Lourenco
    Lorena Mejia-Domenzain
    Erick Moreno-Centeno

