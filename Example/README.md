SuiteSparse/Example: create a package that relies on SuiteSparse

Copyright (c) 2022-2024, Timothy A. Davis, All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause

An example of how to use the SuiteSparse `Find*.cmake` files in cmake
to build a library that depends on SuiteSparse libraries.

    README.md               this file
    License.txt             license
    CMakeLists.txt          primary method for building the package
    Makefile                optional; relies on cmake to do the work
    Include/my.h            created by cmake from Config/my.h.in
    Include/my_internal.h   internal include file
    Config/my.h.in          input file for Include/my.h
    Demo/my_demo.c          C demo program that uses 'my' package
    Demo/my_demo.cc         C++ demo program that uses 'my' package
    Demo/mydemo.out         output of my_demo
    Source/my.c             library source code (C)
    Source/my_cxx.cc        library source code (C++)
    build                   where the 'my' package is built

The 'my' library relies on the following SuiteSparse libraries, each of which
has a cmake module to use in `find_package` that is installed alongside the
compiled libraries:  `AMD`, `BTF`, `CAMD`, `CCOLAMD`, `CHOLMOD`, `COLAMD`,
`CXSparse`, `GPUQREngine`, `GraphBLAS`, `KLU`, `KLU_CHOLMOD`, `LAGraph`, `LDL`,
`Mongoose`, `RBio`, `SPEX`, `SPQR`, `SPQR_CUDA`, `SuiteSparse_GPURuntime`,
`SuiteSparse_config`, and `UMFPACK`.

In addition, the 'my' package relies on the following external libraries:
`BLAS`, `LAPACK`, `OpenMP`, `GMP`, and `MPFR`.

To compile the `my` package and run a demo program:

    First, compile and install all SuiteSparse packages.  Next:

    On Linux or Mac:

        make
        make demos
        sudo make install

    On Windows: load Example/CMakeLists.txt into MS Visual Studio

To remove all compiled files and folders, delete the contents of Example/build
(but keep Example/build/.gitignore):

        make clean

See the instructions in CMakeLists.txt for additional instructions (in
particular, if SuiteSparse is not found).

