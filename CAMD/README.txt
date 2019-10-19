CAMD, Copyright (c) 2007-2013, University of Florida.
Written by Timothy A. Davis (http://www.suitesparse.com), Yanqing Chen, Patrick
R. Amestoy, and Iain S. Duff.  All Rights Reserved.  CAMD is available under
alternate licences; contact T. Davis for details.

CAMD:  a set of routines for permuting sparse matrices prior to
    factorization.  Includes a version in C, a version in Fortran, and a MATLAB
    mexFunction.

Requires SuiteSparse_config, in the ../SuiteSparse_config directory relative to
this directory.

Quick start (Unix, or Windows with Cygwin):

    To compile, test, and install CAMD, you may wish to first configure the
    installation by editting the ../SuiteSparse_config/SuiteSparse_config.mk
    file.  Next, cd to this directory (CAMD) and type "make" (or "make lib" if
    you do not have MATLAB).  When done, type "make clean" to remove unused *.o
    files (keeps the compiled libraries and demo programs).  See the User Guide
    (Doc/CAMD_UserGuide.pdf), or ../SuiteSparse_config/SuiteSparse_config.mk
    for more details.  To install do "make install".

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
    Lib		where the compiled C-callable and Fortran-callable
		CAMD libraries placed.

    ---------------------------------------------------------------------------
    Files in the CAMD directory:
    ---------------------------------------------------------------------------

    Makefile	top-level Makefile.
		Windows users would require Cygwin to use "make"

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

    camd_l_demo.c		C demo program for CAMD ("long" version)
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
    Lib directory:  libcamd.a and libcamd.so library placed here
    ---------------------------------------------------------------------------

    Makefile			Makefile for both shared and static libraries
    old/                        old version of Makefiles

