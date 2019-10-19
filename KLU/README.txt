KLU, Copyright (C) 2004-2013, University of Florida
by Timothy A. Davis and Ekanathan Palamadai.
KLU is also available under other licenses; contact authors for details.
http://www.suitesparse.com

Requires the AMD, COLAMD, and BTF libraries, in ../AMD, ../COLAMD, and ../BTF,
respectively.  Requires the ../SuiteSparse_config/SuiteSparse_config.mk
configuration file.  Optionally uses CHOLMOD (KLU/User example ordering).  The
Tcov tests and the one of the programs in the Demo require CHOLMOD.

To compile the libklu.a library, type "make".  The compiled library is located
in KLU/Lib/libklu.a.  Compile code that uses KLU with -IKLU/Include.
To compile a simple demo (without CHOLMOD), cd to the Demo directory and
type "make klu_simple".

Type "make clean" to remove all but the compiled library, and "make distclean"
to remove all files not in the original distribution.

--------------------------------------------------------------------------------

See KLU/Doc/License.txt for the license.

--------------------------------------------------------------------------------

Files in this distribution:

    Demo                example programs that use KLU (requires CHOLMOD)
    Doc                 documentation
    Include             include files
    Lib                 compiled library
    Makefile            top-level Makefile
    MATLAB              MATLAB interface
    Matrix              test matrices
    README.txt          this file
    Source              source code
    Tcov                exhaustive test of KLU and BTF
    User                example user ordering function (interface to CHOLMOD)

./Demo:
    klu_simple.c        a simple demo (does not require CHOLMOD)
                        compile with "make klu_simple"
    klu_simple.out      output of klu_simple
    kludemo.c           KLU demo (int version)
    kludemo.out         output of "make" in this directory
    kluldemo.c          KLU demo (SuiteSparse_long version)
    Makefile            Makefile for compiling the demo

./Doc:
    ChangeLog
    KLU_UserGuide.bib   Bibiography
    KLU_UserGuide.pdf   PDF version of KLU User Guide
    KLU_UserGuide.tex   TEX source of KLU User Guide
    License.txt         license
    Makefile            Makefile for creating the User Guide
    palamadai_e.pdf     Eka Palamadai's MS thesis

./Include:
    klu.h               user include file
    klu_internal.h      internal include file, not needed by the user
    klu_version.h       internal include file, not needed by the user

./Lib:
    Makefile            Makefile for compiling the KLU C-callable library
                        (with or without CHOLMOD)

./MATLAB:
    Contents.m          list of MATLAB functions in KLU
    klu_demo.m          MATLAB demo
    klu_demo.m.out      output of MATLAB demo (with CHOLMOD)
    klu_install.m       compiles and installs KLU for use in MATLAB, runs demo
    klu.m               MATLAB help for KLU
    klu_make.m          compiles KLU for use in MATLAB
    klu_mex.c           MATLAB mexFunction interface for KLU
    Makefile            Makefile for KLU mexFunction, with CHOLMOD
    Makefile_no_CHOLMOD Makefile for KLU mexFunction, without CHOLMOD
    Test                MATLAB tests

./MATLAB/Test:          KLU tests, requires UFget
    test1.m             
    test2.m
    test3.m
    test4.m
    test5.m

./Matrix:               test matrices for programs in ./Demo and ./Tcov
    1c.mtx
    arrowc.mtx
    arrow.mtx
    ctina.mtx
    GD99_cc.mtx
    impcol_a.mtx
    onec.mtx
    one.mtx
    two.mtx
    w156.mtx

./Source:
    klu_analyze.c       klu_analyze and supporting functions
    klu_analyze_given.c klu_analyze_given and supporting functions
    klu.c               kernel factor/solve functions, not user-callable
    klu_defaults.c      klu_defaults function
    klu_diagnostics.c   klu_rcond, klu_condest, klu_rgrowth, kluflops
    klu_dump.c          debugging functions
    klu_extract.c       klu_extract
    klu_factor.c        klu_factor and supporting functions
    klu_free_numeric.c  klu_free_numeric function
    klu_free_symbolic.c klu_free_symbolic function
    klu_kernel.c        kernel factor functions, not user-callable
    klu_memory.c        klu_malloc, klu_free, klu_realloc, and supporing func.
    klu_refactor.c      klu_refactor function
    klu_scale.c         klu_scale function
    klu_solve.c         klu_solve function
    klu_sort.c          klu_sort and supporting functions
    klu_tsolve.c        klu_tsovle function

./Tcov:                 exhaustive test suite; requires Linux/Unix
    coverage            determine statement coverage
    klultests           KLU SuiteSparse_long tests
    klutest.c           KLU test program
    klutests            KLU int tests
    Makefile            Makefile for compiling and running the tests
    README.txt          README file for Tcov
    vklutests           KLU int tests, using valgrind
    vklultests          KLU SuiteSparse_long tests, using valgrind

./User:
    klu_cholmod.c       sample KLU user ordering function (int version)
    klu_cholmod.h       include file for klu_cholmod and klu_l_cholmod
    klu_l_cholmod.c     sample KLU user ordering function (SuiteSparse_long) 
    Makefile            Makefile for compiling the user ordering functions
    README.txt          README for User directory
