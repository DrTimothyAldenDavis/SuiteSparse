SuiteSparse/Example: create a package that relies on SuiteSparse

Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause

An example of how to use the SuiteSparse `Find*.cmake` files in cmake
to build a library that depends on SuiteSparse libraries.

    README.md           this file
    CMakeLists.txt      primary method for building the package
    Makefile            optional; relies on cmake to do the work
    Include/my.h        created by cmake from Config/my.h.in
    Config/my.h.in      input file for Include/my.h
    Demo/my_demo.c      demo program that uses 'my' package
    Demo/mydemo.out     output of my_demo
    Source/my.c         library source code
    build               where the 'my' package is built
    cmake_modules/FindGMP.cmake     how to find the GMP library
    cmake_modules/FindMPFR.cmake    how to find the MPFR library

The 'my' library relies on the following SuiteSparse libraries, each of which
has a cmake module to use in `find_package` that is installed alongside the
compiled libraries (/usr/local/lib/cmake/SuiteSparse) `AMD`, `BTF`, `CAMD`,
`CCOLAMD`, `CHOLMOD`, `CHOLMOD_CUDA`, `COLAMD`, `CXSparse`, `GPUQREngine`,
`GraphBLAS`, `KLU`, `KLU_CHOLMOD`, `LDL`, `Mongoose`, `RBio`, `SPEX`, `SPQR`,
`SPQR_CUDA`, `SuiteSparse_GPURuntime`, `SuiteSparse_config`, and `UMFPACK`.

In addition, the 'my' package relies on the following external libraries:
`BLAS`, `LAPACK`, `OpenMP`, `GMP`, and `MPFR`.  The latter two (GMP and MPFR)
do not have cmake `find_package` modules.  These can be found in
`SuiteSparse/Example/cmake_modules`, or in `SuiteSparse/SPEX/cmake_modules`.
They are not installed in /usr/local/lib/cmake/SuiteSparse.

To compile the `my` package and run a demo program:

    On Linux or Mac:

        make
        make demos
        sudo make install

    On Windows: load Example/CMakeLists.txt into MS Visual Studio

To remove all compiled files and folders, delete the contents of Example/build
(but keep Example/build/.gitignore):

        make clean

