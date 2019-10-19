CXSparse: a Concise Sparse Matrix package - Extended.
Version 2.0.2, Copyright (c) 2006, Timothy A. Davis, Aug 23, 2006.
Derived from CSparse.  Conversion originally by David Bateman, Motorola,
and then modified by Tim Davis.  ANSI C99 is required, with support for
the _Complex data type.

CXSparse is a version of CSparse that operates on both real and complex
matrices, using either int or UF_long integers.  A UF_long is normally
just a long on most platforms, but becomes __int64 on WIN64.

Refer to "Direct Methods for Sparse Linear Systems," Timothy A. Davis,
SIAM, Philaelphia, 2006.  No detailed user guide is included in this
package; the user guide is the book itself.

To compile the C-library (./Source), C demo programs (./Demo) just type "make"
in this directory.  To run the exhaustive statement coverage tests, type
"make" in the Tcov directory; the Tcov tests assume you are using Linux.  To
remove all files not in the original distribution, type "make distclean".  No
MATLAB interface is provided.  I recommend that you use a different level of
optimization than "cc -O", which was chosen so that the Makefile is portable.
See Source/Makefile.

This package is backward compatible with CSparse.  That is, user code that
uses CSparse may switch to using CXSparse without any changes to the user code.
Each CXSparse function has a generic version with the same name as the CSparse
function, and four type-specific versions.  For example:

    cs_add	same as cs_add_di by default, but can be changed to use UF_long
		integers if user code is compiled with -DCS_LONG, and/or can
		be changed to operate on complex matrices with -DCS_COMPLEX.

    cs_di_add	double/int version of cs_add
    cs_dl_add	double/UF_long version of cs_add
    cs_ci_add	complex/int version of cs_add
    cs_cl_add	complex/UF_long version of cs_add

The sparse matrix data structures are treated in the same way:  cs, css,
csn, and csd become cs_di, cs_dis, cs_din, and cs_did for the double/int case,
cs_cl, cs_cls, cs_cln, and cs_cld for the complex/UF_long case, and so on.

See cs_demo.c for a type-generic user program, and cs_cl_demo.c for a
type-specific version of the same program (complex/UF_long).

Several macros are available in CXSparse (but not in CSparse) to allow user
code to be written in a type-generic manner:

    CS_INT	int by default, UF_long if -DCS_LONG compiler flag is used
    CS_ENTRY	double by default, double complex if -DCS_COMPLEX flag is used.
    CS_ID	"%d" or "%ld", for printf and scanf of the CS_INT type.
    CS_INT_MAX	INT_MAX or LONG_MAX, the largest possible value of CS_INT.
    CS_REAL(x)	x or creal(x)
    CS_IMAG(x)	0 or cimag(x)
    CS_CONJ(x)	x or conj(x)
    CS_ABS(x)	fabs(x) or cabs(x)

Even the name of the include file (cs.h) is the same.  To use CXSparse instead
of CSparse, simply compile with -ICXSparse/Source instead of -ICSparse/Source,
and link against libcxsparse.a instead of the CSparse libcsparse.a library.

To determine at compile time if CXSparse or CSparse is being used:

    #ifdef CXSPARSE
	CXSparse is in use.  The generic functions equivalent to CSparse may
	be used (cs_add, etc).  These generic functions can use different
	types, depending on the -DCS_LONG and -DCS_COMPLEX compile flags, with
	the default being double/int.  The type-specific functions and data
	types (cs_di_add, cs_di, CS_INT, etc.) can be used.
    #else
	CSparse is in use.  Only the generic functions "cs_add", etc., are
	available, and they are of type double/int.
    #endif

See cs.h for the prototypes of each function, and the book "Direct Methods
for Sparse Linear Systems" for full documentation of CSparse and CXSparse.

Other changes from CSparse:  cs_transpose performs the complex conjugate
transpose if values>0 (C=A'), the pattern-only transpose if values=0
(C=spones(A') in MATLAB), and the array transpose if values<0 (C=A.' in
MATLAB notation).  A set of four conversion routines are included in CXSparse,
to convert real matrices to/from complex matrices.

CXSparse is generated automatically from CSparse.  Refer to
http://www.cise.ufl.edu/research/sparse/CSparse for details.

--------------------------------------------------------------------------------
Contents:
--------------------------------------------------------------------------------

ChangeLog	Change Log for CSparse (most but not all apply to CXSparse)
Demo/		demo C programs that use CXSparse
lesser.txt	the GNU LGPL
License.txt	license (GNU LGPL)
Makefile	Makefile for the whole package
Matrix/		sample matrices (with extra complex matrices CXSparse)
README.txt	this file
Source/		primary CXSparse source files
Tcov/		CXSparse tests


--------------------------------------------------------------------------------
./Source:	Primary source code for CXSparse
--------------------------------------------------------------------------------

cs_add.c	add sparse matrices
cs_amd.c	approximate minimum degree
cs_chol.c	sparse Cholesky
cs_cholsol.c	x=A\b using sparse Cholesky
cs_compress.c	convert a compress form to compressed-column form
cs_counts.c	column counts for Cholesky and QR
cs_convert.c	convert real to complex and complex to real (not in CSparse)
cs_cumsum.c	cumulative sum
cs_dfs.c	depth-first-search
cs_dmperm.c	Dulmage-Mendelsohn permutation
cs_droptol.c	drop small entries from a sparse matrix
cs_dropzeros.c	drop zeros from a sparse matrix
cs_dupl.c	remove (and sum) duplicates
cs_entry.c	add an entry to a triplet matrix
cs_ereach.c	nonzero pattern of Cholesky L(k,:) from etree and triu(A(:,k))
cs_etree.c	find elimination tree
cs_fkeep.c	drop entries from a sparse matrix
cs_gaxpy.c	sparse matrix times dense matrix
cs.h		include file for CXSparse
cs_happly.c	apply Householder reflection
cs_house.c	Householder reflection (*** NOTE: different algo. from CSparse)
cs_ipvec.c	x(p)=b
cs_leaf.c	determine if j is a leaf of the skeleton matrix and find lca
cs_load.c	load a sparse matrix from a file
cs_lsolve.c	x=L\b
cs_ltsolve.c	x=L'\b
cs_lu.c		sparse LU factorization
cs_lusol.c	x=A\b using sparse LU factorization
cs_malloc.c	memory manager
cs_maxtrans.c	maximum transveral (permutation for zero-free diagonal)
cs_multiply.c	sparse matrix multiply
cs_norm.c	sparse matrix norm
cs_permute.c	permute a sparse matrix
cs_pinv.c	invert a permutation vector
cs_post.c	postorder an elimination tree
cs_print.c	print a sparse matrix
cs_pvec.c	x=b(p)
cs_qr.c		sparse QR
cs_qrsol.c	solve a least-squares problem
cs_randperm.c	random permutation
cs_reach.c	find nonzero pattern of x=L\b for sparse L and b
cs_scatter.c	scatter a sparse vector
cs_scc.c	strongly-connected components
cs_schol.c	symbolic Cholesky
cs_spsolve.c	x=Z\b where Z, x, and b are sparse, and Z upper/lower triangular
cs_sqr.c	symbolic QR (also can be used for LU)
cs_symperm.c	symmetric permutation of a sparse matrix
cs_tdfs.c	depth-first-search of a tree
cs_transpose.c	transpose a sparse matrix
cs_updown.c	sparse rank-1 Cholesky update/downate
cs_usolve.c	x=U\b
cs_util.c	various utilities (allocate/free matrices, workspace, etc)
cs_utsolve.c	x=U'\b
Makefile	Makefile for CXSparse
README.txt	README file for CXSparse

--------------------------------------------------------------------------------
./Demo:		C program demos
--------------------------------------------------------------------------------

cs_ci_demo1.c	complex/int version of cs_demo1.c
cs_ci_demo2.c	complex/int version of cs_demo2.c
cs_ci_demo3.c	complex/int version of cs_demo3.c
cs_ci_demo.c	complex/int version of cs_demo.c
cs_ci_demo.h	complex/int version of cs_demo.h

cs_cl_demo1.c	complex/UF_long version of cs_demo1.c
cs_cl_demo2.c	complex/UF_long version of cs_demo2.c
cs_cl_demo3.c	complex/UF_long version of cs_demo3.c
cs_cl_demo.c	complex/UF_long version of cs_demo.c
cs_cl_demo.h	complex/UF_long version of cs_demo.h

cs_demo1.c	read a matrix from a file and perform basic matrix operations
cs_demo2.c	read a matrix from a file and solve a linear system
cs_demo3.c	read a matrix, solve a linear system, update/downdate
cs_demo.c	support routines for cs_demo*.c
cs_demo.h	include file for demo programs

cs_demo.out	output of "make", which runs the demos on some matrices

cs_di_demo1.c	double/int version of cs_demo1.c
cs_di_demo2.c	double/int version of cs_demo2.c
cs_di_demo3.c	double/int version of cs_demo3.c
cs_di_demo.c	double/int version of cs_demo.c
cs_di_demo.h	double/int version of cs_demo.h

cs_dl_demo1.c	double/UF_long version of cs_demo1.c
cs_dl_demo2.c	double/UF_long version of cs_demo2.c
cs_dl_demo3.c	double/UF_long version of cs_demo3.c
cs_dl_demo.c	double/UF_long version of cs_demo.c
cs_dl_demo.h	double/UF_long version of cs_demo.h

cs_idemo.c	convert real matrices to/from complex (int version)
cs_ldemo.c	convert real matrices to/from complex (UF_long version)

Makefile	Makefile for Demo programs
readhb.f	read a Rutherford-Boeing matrix (real matrices only)
README.txt	Demo README file

--------------------------------------------------------------------------------
./Matrix:	    Sample matrices, most from Rutherford/Boeing collection
--------------------------------------------------------------------------------

ash219		    overdetermined pattern of Holland survey.  Ashkenazi, 1974.
bcsstk01	    stiffness matrix for small generalized eigenvalue problem
bcsstk16	    stiffness matrix, Corp of Engineers dam
fs_183_1	    unsymmetric facsimile convergence matrix
lp_afiro	    NETLIB afiro linear programming problem
mbeacxc		    US economy, 1972.  Dan Szyld, while at NYU
t1		    small example used in Chapter 2
west0067	    Cavett problem with 5 components (chemical eng., Westerberg)

c_mbeacxc	    complex version of mbeacxc
c_west0067	    complex version of west0067
mhd1280b	    Alfven spectra in magnetohydrodynamics (complex)
neumann		    complex matrix
qc324		    model of H+ in an electromagnetic field (complex)
t2		    small complex matrix
t3		    small complex matrix
t4		    small complex matrix
young1c		    aeronautical problem (complex matrix)

--------------------------------------------------------------------------------
./Tcov:		    Exhaustive test coverage of CXSparse
--------------------------------------------------------------------------------

covall		    same as covall.linux
covall.linux	    find coverage (Linux)
covall.sol	    find coverage (Solaris)
cov.awk		    coverage summary
cover		    print uncovered lines
covs		    print uncovered lines
cstcov_malloc_test.c    malloc test
cstcov_malloc_test.h
cstcov_test.c	    main program for Tcov tests
gcovs		    run gcov (Linux)
Makefile	    Makefile for Tcov tests
nil		    an empty matrix
zero		    a 1-by-1 zero matrix
czero		    a 1-by-1 complex zero matrix
README.txt	    README file for Tcov directory

--------------------------------------------------------------------------------
Change Log:
--------------------------------------------------------------------------------

May 5, 2006.  Version 2.0.1 released.

    * long changed to UF_long, dependency in ../UFconfig/UFconfig.h added.
	"UF_long" is a #define'd term in UFconfig.h.  It is normally defined
	as "long", but can be redefined as something else if desired.
	On Windows-64, it becomes __int64.

Mar 6, 2006

    "double complex" changed to "double _Complex", to avoid conflicts when
    CXSparse is compiled with a C++ compiler.  Other minor changes to cs.h.

Refer to CSparse for changes in CSparse, which are immediately propagated
into CXSparse (those Change Log entries are not repeated here).

