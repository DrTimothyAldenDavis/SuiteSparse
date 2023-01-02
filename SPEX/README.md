SPEX is a software package for SParse EXact algebra

Files and folders in this distribution:

    README.md       this file
    
                    
    SPEX_Left_LU    Sparse left-looking integer-preserving
                    LU factorization for exactly solve 
                    sparse linear systems (Release)
   
    SPEX_UTIL       Utility functions for all SPEX components
    
    Makefile        compiles SPEX and its dependencies

Dependencies:

    AMD                 approximate minimum degree ordering
    
    COLAMD              column approximate minimum degree ordering
    
    SuiteSparse_config  configuration for all of SuiteSparse
    
    GNU GMP             GNU Multiple Precision Arithmetic Library 
                        for big integer operations
    
    GNU MPFR            GNU Multiple Precision Floating-Point Reliable
                        Library for arbitrary precision floating point
                        operations

Default instalation locations:

    include
    lib
    share
    
To compile SPEX and its dependencies, just type "make" in this folder.
To run a few short demos, type "make demos"
To install the package system-wide, type "sudo make install".

Primary Author: Chris Lourenco

Coauthors (alphabetical order):

    Jinhao Chen
    Tim Davis    
    Erick Moreno-Centeno

NOTE: The cmake build system for SPEX 2.0.3 in SuiteSparse v7.0.0
works on Windows, but (so far) only when using the MSYS2 build
system.  It doesn't yet work in MS Visual Studio, because the GMP
library isn't available on Windows for MSVC (just msys2 and
cygwin).  We hope to resolve this issue in the future.
