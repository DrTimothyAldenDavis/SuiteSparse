CAMD, Copyright (c) 2007-2022, Timothy A. Davis, Yanqing Chen, Patrick R.
Amestoy, and Iain S. Duff.  All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause

CAMD:  a set of routines for permuting sparse matrices prior to
    factorization.  Includes a version in C, a version in Fortran, and a MATLAB
    mexFunction.

Requires SuiteSparse_config, in the ../SuiteSparse_config directory relative to
this directory.

Quick start (Linux or MacOSX):

    To compile and install the library for system-wide usage:

        make
        sudo make install

    To compile/install for local usage (SuiteSparse/lib and SuiteSparse/include)

        make local
        make install

    To run the demos

        make demos

Quick start (for MATLAB users);

    To compile, test, and install the CAMD mexFunction, cd to the
    CAMD/MATLAB directory and type camd_make at the MATLAB prompt.

-------------------------------------------------------------------------------

CAMD License: refer to CAMD/Doc/License.txt

Availability:

    http://www.suitesparse.com

-------------------------------------------------------------------------------

This is the CAMD README file.  It is a terse overview of CAMD.
Refer to the User Guide (Doc/CAMD_UserGuide.pdf) for how to install
and use CAMD.

Description:

    CAMD is a set of routines for pre-ordering sparse matrices prior to Cholesky
    or LU factorization, using the approximate minimum degree ordering
    algorithm with optional ordering constraints.  Written in ANSI/ISO C with
    a MATLAB interface.

Authors:

    Timothy A. Davis (DrTimothyAldenDavis@gmail.com)
    Patrick R. Amestory, ENSEEIHT, Toulouse, France.
    Iain S. Duff, Rutherford Appleton Laboratory, UK.

Acknowledgements:

    This work was supported by the National Science Foundation, under
    grants DMS-9504974, DMS-9803599, and CCR-0203270.

    Portions of this work were done while on sabbatical at Stanford University
    and Lawrence Berkeley National Laboratory (with funding from the SciDAC
    program).  I would like to thank Gene Golub, Esmond Ng, and Horst Simon
    for making this sabbatical possible.

-------------------------------------------------------------------------------
Files and directories in the CAMD distribution:
-------------------------------------------------------------------------------

    ---------------------------------------------------------------------------
    Subdirectories of the CAMD directory:
    ---------------------------------------------------------------------------

    Doc		documentation
    Source	primary source code
    Include	include file for use in your code that calls CAMD
    Demo	demo programs.  also serves as test of the CAMD installation.
    MATLAB	CAMD mexFunction for MATLAB, and supporting m-files
    build       where the compiled libraries and demos are placed
    Config      source file to construct camd.h

    ---------------------------------------------------------------------------
    Files in the CAMD directory:
    ---------------------------------------------------------------------------

    Makefile	a very simple Makefile (optional); just for simplifying cmake
    CMakeLists.txt  cmake script for building CAMD

    README.txt	this file

    ---------------------------------------------------------------------------
    Doc directory: documentation
    ---------------------------------------------------------------------------

    ChangeLog			change log
    License.txt			the CAMD License
    Makefile			for creating the documentation
    CAMD_UserGuide.bib		CAMD User Guide (references)
    CAMD_UserGuide.tex		CAMD User Guide (LaTeX)
    CAMD_UserGuide.pdf		CAMD User Guide (PDF)

    docdiff			tools for comparing CAMD with AMD
    cdiff
    camd.sed

    ---------------------------------------------------------------------------
    Source directory:
    ---------------------------------------------------------------------------

    camd_order.c		user-callable, primary CAMD ordering routine
    camd_control.c		user-callable, prints the control parameters
    camd_defaults.c		user-callable, sets default control parameters
    camd_info.c			user-callable, prints the statistics from CAMD

    camd_1.c			non-user-callable, construct A+A'
    camd_2.c			user-callable, primary ordering kernel
				(a C version of camd.f and camdbar.f, with
				post-ordering added)
    camd_aat.c			non-user-callable, computes nnz (A+A')
    camd_dump.c			non-user-callable, debugging routines
    camd_postorder.c		non-user-callable, postorder
    camd_valid.c		non-user-callable, verifies a matrix
    camd_preprocess.c		non-user-callable, computes A', removes duplic

    camd_l*                     same as above, but with int64_t integers

    ---------------------------------------------------------------------------
    Include directory:
    ---------------------------------------------------------------------------

    camd.h			include file for C programs that use CAMD
    camd_internal.h		non-user-callable, include file for CAMD

    ---------------------------------------------------------------------------
    Demo directory:
    ---------------------------------------------------------------------------

    Makefile			to compile the demos

    camd_demo.c			C demo program for CAMD
    camd_demo.out		output of camd_demo.c

    camd_demo2.c		C demo program for CAMD, jumbled matrix
    camd_demo2.out		output of camd_demo2.c

    camd_l_demo.c		C demo program for CAMD (int64_t version)
    camd_l_demo.out		output of camd_l_demo.c

    camd_simple.c		simple C demo program for CAMD
    camd_simple.out		output of camd_simple.c

    ---------------------------------------------------------------------------
    MATLAB directory:
    ---------------------------------------------------------------------------

    Contents.m			for "help camd" listing of toolbox contents

    camd.m			MATLAB help file for CAMD
    camd_make.m			MATLAB m-file for compiling CAMD mexFunction
    camd_install.m		compile and install CAMD mexFunctions

    camd_mex.c			CAMD mexFunction for MATLAB

    camd_demo.m			MATLAB demo for CAMD
    camd_demo.m.out		diary output of camd_demo.m
    can_24.mat			input file for CAMD demo

    ---------------------------------------------------------------------------
    build directory:  libcamd.a and libcamd.so library placed here
    ---------------------------------------------------------------------------

    .gitignore                  only file in the original distribution

