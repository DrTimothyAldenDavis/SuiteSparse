CCOLAMD version 2.5: constrained column approximate minimum degree ordering
Copyright (C) 2005, Univ. of Florida.  Authors: Timothy A. Davis,
Sivasankaran Rajamanickam, and Stefan Larimore.  Closely based on COLAMD by
Davis, Stefan Larimore, in collaboration with Esmond Ng, and John Gilbert.
http://www.cise.ufl.edu/research/sparse
-------------------------------------------------------------------------------

The CCOLAMD column approximate minimum degree ordering algorithm computes
a permutation vector P such that the LU factorization of A (:,P)
tends to be sparser than that of A.  The Cholesky factorization of
(A (:,P))'*(A (:,P)) will also tend to be sparser than that of A'*A.
CSYMAMD is a symmetric minimum degree ordering method based on CCOLAMD, also
available as a MATLAB-callable function.  It constructs a matrix M such
that M'*M has the same pattern as A, and then uses CCOLAMD to compute a column
ordering of M.

Requires UFconfig, in the ../UFconfig directory relative to this directory.

To compile and test the colamd m-files and mexFunctions, just unpack the
CCOLAMD/ directory from the CCOLAMD.tar.gz file, and run MATLAB from
within that directory.  Next, type ccolamd_test to compile and test ccolamd
and csymamd.  This will work on any computer with MATLAB (Unix, PC, or Mac).
Alternatively, type "make" (in Unix) to compile and run a simple example C
code, and to compile the C-callable library (libccolamd.a).

Other "make" targets:

    make matlab		compiles MATLAB mexFunctions only
    make libccolamd.a	compiles a C-callable library containing ccolamd
    make clean		removes all files not in the distribution, except for
			libccolamd.a
    make distclean	removes all files not in the distribution

To use ccolamd and csymamd within an application written in C, all you need are
colamd.c and colamd.h, which are the C-callable ccolamd/csymamd codes.
See ccolamd.c for more information on how to call ccolamd from a C program.
It contains a complete description of the C-interface to CCOLAMD and CSYMAMD.

	Copyright (c) 1998-2005 by the University of Florida.
	All Rights Reserved.

	Licensed under the GNU LESSER GENERAL PUBLIC LICENSE.

-------------------------------------------------------------------------------

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

-------------------------------------------------------------------------------


Related papers:

	T. A. Davis, W. W. Hager, and S. Rajamanickam, Multiple-rank updates
	to a supernodal sparse Cholesky factorization, under preparation.

	T. A. Davis, W. W. Hager, and S. Rajamanickam, CHOLMOD: a sparse
	Cholesky update/downdate package, under preparation.  CHOLMOD's
	nested dissection ordering relies on CCOLAMD and CSYMAMD to order
	the matrix after graph partitioning is used to find the ordering
	constraints.

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

Authors:  Timothy A. Davis, Sivasankaran Rajamanickam, and Stefan Larimore.
	Closely based on COLAMD by Stefan I. Larimore and Timothy A. Davis,
	University of Florida, in collaboration with John Gilbert, Xerox PARC
	(now at UC Santa Barbara), and Esmong Ng, Lawrence Berkeley National
	Laboratory (much of this work he did while at Oak Ridge National
	Laboratory). 

COLAMD files:

	CCOLAMD.tar.gz:
		All files, as a gzipped, Unix tar file.
		The *.m, and *mex.c files are for use in MATLAB.

	ccolamd.c: the primary ccolamd computational kernel.

	ccolamd.h: include file for ccolamd/csymamd library.

	ccolamd.m: the MATLAB interface to ccolamd.

	ccolamd_demo.m: MATLAB demo file for ccolamd and csymamd
		(also compiles the ccolamdmex and csymamdmex mexFunctions).

	ccolamdmex.c: ccolamd mexFunction for use in MATLAB.

	ccolamd_example.c: example C main program that calls ccolamd and csymamd

	ccolamd_example.out: output of ccolamd_example.c.

	Makefile: Makefile for ccolamd_example.c

	csymamd.m: the MATLAB interface to csymamd.

	csymamdmex.c: csymamd mexFunction for use in MATLAB.

	README:  this file

	ChangeLog: a log of changes since Version 1.0.

	ccolamd_test.m:	test code

	ccolamdtestmex.c:  test code

	luflops.m:  test code

	csymamdtestmex.c:  test code

	lesser.txt: a verbatim copy of the GNU LGPL Version 2.1 license
