The COLAMD ordering method - Version 2.6
-------------------------------------------------------------------------------

The COLAMD column approximate minimum degree ordering algorithm computes
a permutation vector P such that the LU factorization of A (:,P)
tends to be sparser than that of A.  The Cholesky factorization of
(A (:,P))'*(A (:,P)) will also tend to be sparser than that of A'*A.
SYMAMD is a symmetric minimum degree ordering method based on COLAMD,
available as a MATLAB-callable function.  It constructs a matrix M such
that M'*M has the same pattern as A, and then uses COLAMD to compute a column
ordering of M.  Colamd and symamd tend to be faster and generate better
orderings than their MATLAB counterparts, colmmd and symmmd.

To compile and test the colamd m-files and mexFunctions, just unpack the
COLAMD/ directory from the COLAMD.tar.gz file, and run MATLAB from
within that directory.  Next, type colamd_test to compile and test colamd
and symamd.  This will work on any computer with MATLAB (Unix, PC, or Mac).
Alternatively, type "make" (in Unix) to compile and run a simple example C
code, without using MATLAB.

Colamd is a built-in routine in MATLAB, available from The 
Mathworks, Inc.  Under most cases, the compiled COLAMD from Versions 2.0
through 2.6 do not differ.  Colamd Versions 2.2 and 2.3 differ only in their
mexFunction interaces to MATLAB.  v2.4 fixes a bug in the symamd routine in
v2.3.  The bug (in v2.3 and earlier) has no effect on the MATLAB symamd
mexFunction.  v2.5 adds additional checks for integer overflow, so that
the "int" version can be safely used with 64-bit pointers.  Refer to the
ChangeLog for more details.

    NOTE: DO NOT ATTEMPT TO USE THIS CODE IN 64-BIT MATLAB (v7.3).
    It is not yet ported to that version of MATLAB.

To use colamd and symamd within an application written in C, all you need are
colamd.c, colamd_global.c, and colamd.h, which are the C-callable
colamd/symamd codes.  See colamd.c for more information on how to call
colamd from a C program.

Requires UFconfig, in the ../UFconfig directory relative to this directory.

	Copyright (c) 1998-2006, Timothy A. Davis, All Rights Reserved.

	See http://www.cise.ufl.edu/research/sparse/colamd (the colamd.c
	file) for the License.


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
	1998.  CISE Tech Report TR-98-016.  Available at 
	ftp://ftp.cise.ufl.edu/cis/tech-reports/tr98/tr98-016.ps
	via anonymous ftp.

	Approximate Deficiency for Ordering the Columns of a Matrix,
	J. L. Kern, Senior Thesis, Dept. of Computer and Information
	Science and Engineering, University of Florida, Gainesville, FL,
	1999.  Available at http://www.cise.ufl.edu/~davis/Kern/kern.ps 


Authors:  Stefan I. Larimore and Timothy A. Davis, University of Florida,
in collaboration with John Gilbert, Xerox PARC (now at UC Santa Barbara),
and Esmong Ng, Lawrence Berkeley National Laboratory (much of this work
he did while at Oak Ridge National Laboratory). 

COLAMD files

	colamd.c: the primary colamd computational kernel.

	colamd.h: include file for colamd/symamd library.

	colamd.m: the MATLAB interface to colamd.

	colamd_demo.m: MATLAB demo file for colamd and symamd
		(also compiles the colamdmex and symamdmex mexFunctions).

	colamdmex.c: colamd mexFunction for use in MATLAB.

	colamd_example.c: example C main program that calls colamd and symamd.

	colamd_example.out: output of colamd_example.c.

	Makefile: Makefile for colamd_example.c

	symamd.m: the MATLAB interface to symamd.

	symamdmex.c: symamd mexFunction for use in MATLAB.

	README:  this file

	ChangeLog: a log of changes since Version 1.0.

	colamd_test.m:	test code

	colamdtestmex.c:  test code

	luflops.m:  test code

	symamdtestmex.c:  test code
