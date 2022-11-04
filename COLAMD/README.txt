COLAMD, Copyright 1998-2022, Timothy A. Davis.  http://www.suitesparse.com
-------------------------------------------------------------------------------

SPDX-License-Identifier: BSD-3-clause

The COLAMD column approximate minimum degree ordering algorithm computes
a permutation vector P such that the LU factorization of A (:,P)
tends to be sparser than that of A.  The Cholesky factorization of
(A (:,P))'*(A (:,P)) will also tend to be sparser than that of A'*A.
SYMAMD is a symmetric minimum degree ordering method based on COLAMD,
available as a MATLAB-callable function.  It constructs a matrix M such
that M'*M has the same pattern as A, and then uses COLAMD to compute a column
ordering of M.  Colamd and symamd tend to be faster and generate better
orderings than their MATLAB counterparts, colmmd and symmmd.

To compile and install the colamd m-files and mexFunctions, just cd to
COLAMD/MATLAB and type colamd_install in the MATLAB command window.
A short demo will run.  Optionally, type colamd_test to run an extensive tests.

Colamd is a built-in routine in MATLAB, available from The 
Mathworks, Inc.  Under most cases, the compiled COLAMD from Versions 2.0 to the
current version do not differ.  Colamd Versions 2.2 and 2.3 differ only in their
mexFunction interaces to MATLAB.  v2.4 fixes a bug in the symamd routine in
v2.3.  The bug (in v2.3 and earlier) has no effect on the MATLAB symamd
mexFunction.  v2.5 adds additional checks for integer overflow, so that
the "int" version can be safely used with 64-bit pointers.  Refer to the
ChangeLog for more details.

COLAMD includes a simple top-level Makefile, which is optional.  All the work
is done via cmake.  Windows users can simply import the CMakeLists.txt into MS
Visual Studio.

"make" targets:

    make                compiles the COLAMD library;
                            "make install" will install in /usr/local/lib,
                            /usr/local/include, SuiteSparse/lib, and
                            SuiteSparse/include
    make demos          compiles and runs a few demos
    make library	compiles a C-callable library containing colamd
    make clean		removes all files not in the distribution,
                            but keeps the compiled libraries.
    make distclean	removes all files not in the distribution
    make local          compiles the COLAMD library;
                            "make install" will install only in
                            SuiteSparse/lib and SuiteSparse/include
    make install        installs the library
    make uninstall      uninstalls the library

To use colamd and symamd within an application written in C, all you need are
colamd.c, and colamd.h, which are the C-callable
colamd/symamd codes.  See colamd.c for more information on how to call
colamd from a C program.

Requires SuiteSparse_config, in the ../SuiteSparse_config directory relative to
this directory.

See COLAMD/Doc/License.txt for the License.

Related papers:

	T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, An approximate column
	minimum degree ordering algorithm, ACM Transactions on Mathematical
	Software, vol. 30, no. 3., pp. 353-376, 2004.

	T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836: COLAMD,
	an approximate column minimum degree ordering algorithm, ACM
	Transactions on Mathematical Software, vol. 30, no. 3., pp. 377-380,
	2004.

	"An approximate minimum degree column ordering algorithm",
	S. I. Larimore, MS Thesis, Dept. of Computer and Information
	Science and Engineering, University of Florida, Gainesville, FL,
	1998.  CISE Tech Report TR-98-016.

	Approximate Deficiency for Ordering the Columns of a Matrix,
	J. L. Kern, Senior Thesis, Dept. of Computer and Information
	Science and Engineering, University of Florida, Gainesville, FL,
	1999.


Authors:  Stefan I. Larimore and Timothy A. Davis,
in collaboration with John Gilbert, Xerox PARC (now at UC Santa Barbara),
and Esmong Ng, Lawrence Berkeley National Laboratory (much of this work
he did while at Oak Ridge National Laboratory). 

COLAMD files:

    Demo	    simple demo
    Doc		    additional documentation (see colamd.c for more)
    Include	    include file
    Config          source for colamd.h
    Makefile	    optional Makefile
    CMakeLists.txt  for using cmake to build COLAMD
    MATLAB	    MATLAB functions
    README.txt	    this file
    Source	    C source code

    ./Demo:
    colamd_example.c	    simple example
    colamd_example.out	    output of colamd_example.c
    colamd_l_example.c	    simple example, long integers
    colamd_l_example.out    output of colamd_l_example.c

    ./Doc:
    ChangeLog	    change log
    License.txt     license

    ./Include:
    colamd.h	    include file

    ./MATLAB:
    colamd2.m		MATLAB interface for colamd2
    colamd_demo.m	simple demo
    colamd_install.m	compile and install colamd2 and symamd2
    colamd_make.m	compile colamd2 and symamd2
    colamdmex.ca	MATLAB mexFunction for colamd2
    colamd_test.m	extensive test
    colamdtestmex.c	test function for colamd
    Contents.m		contents of the MATLAB directory
    luflops.m		test code
    Makefile		Makefile for MATLAB functions
    symamd2.m		MATLAB interface for symamd2
    symamdmex.c		MATLAB mexFunction for symamd2
    symamdtestmex.c	test function for symamd

    ./Source:
    colamd.c		primary source code
    colamd_l.c		primary source code for int64_t version

    ./build:            where COLAMD is built
    .gitignore

