CSparse: a Concise Sparse Matrix package.
Version 2.0.2, Copyright (c) 2006, Timothy A. Davis, Aug 23, 2006.

Refer to "Direct Methods for Sparse Linear Systems," Timothy A. Davis,
SIAM, Philaelphia, 2006.  No detailed user guide is included in this
package; the user guide is the book itself.

The algorithms contained in CSparse have been chosen with five goals in mind:
(1) they must embody much of the theory behind sparse matrix algorithms,
(2) they must be either asymptotically optimal in their run time and memory
    usage or be fast in practice,
(3) they must be concise so as to be easily understood and short enough to
    print in the book,
(4) they must cover a wide spectrum of matrix operations, and
(5) they must be accurate and robust.
The focus is on direct methods; iterative methods and solvers for
eigenvalue problems are beyond the scope of this package.

No detailed user guide is included in this package; the user guide is the
book itself.  Some indication of how to call the CSparse C routines is given
by the M-files in the MATLAB/CSparse directory.

Complex matrices are not supported, except for methods that operate only
on the nonzero pattern of a matrix.  A complex version of CSparse appears
as a separate package, CXSparse ("Concise Extended Sparse matrix package").

The performance of the sparse factorization methods in CSparse will not be
competitive with UMFPACK or CHOLMOD, but the codes are much more concise and
easy to understand (see the above goals).  Other methods are competitive.

Some of the MATLAB tests require the AMD package.
See http://www.cise.ufl.edu/research/sparse for CSparse and the AMD ordering
package.  See the ./License.txt file for the license (GNU LGPL).

To compile the C-library (./Source) and C demo programs (./Demo) just type
"make" in this directory.  This should work on any system with the "make"
command.  To run the exhaustive tests, type "make" in the Tcov directory
(Linux is assumed).  To compile the MATLAB mexFunctions type "make mex" in
this directory, or just "make" in the MATLAB directory.  To remove all files
not in the original distribution, type "make distclean".  I recommend that you
use a different level of optimization than "cc -O", which was chosen so that
the Makefile is portable.  See Source/Makefile.

You can simply type "cs_install" while in the CSparse/MATLAB directory to
compile and install CSparse for use in MATLAB.  This is especially useful for
a typical Microsoft Windows system, which does not include "make".  For more
details, see CSparse/MATLAB/README.txt.

--------------------------------------------------------------------------------
Contents:
--------------------------------------------------------------------------------

ChangeLog	changes in CSparse since first release
Demo/		demo C programs that use CSparse
lesser.txt	the GNU LGPL
License.txt	license (GNU LGPL)
Makefile	Makefile for the whole package
MATLAB/		MATLAB interface, demos, and tests for CSparse
Matrix/		sample matrices
README.txt	this file
Source/		primary CSparse source files (C only, no MATLAB)
Tcov/		CSparse tests

--------------------------------------------------------------------------------
./Source:	Primary source code for CSparse
--------------------------------------------------------------------------------

cs_add.c	add sparse matrices
cs_amd.c	approximate minimum degree
cs_chol.c	sparse Cholesky
cs_cholsol.c	x=A\b using sparse Cholesky
cs_compress.c	convert a triplet form to compressed-column form
cs_counts.c	column counts for Cholesky and QR
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
cs.h		include file for CSparse
cs_happly.c	apply Householder reflection
cs_house.c	compute Householder reflection
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
cs_spsolve.c	x=G\b where G, x, and b are sparse, and G upper/lower triangular
cs_sqr.c	symbolic QR (also can be used for LU)
cs_symperm.c	symmetric permutation of a sparse matrix
cs_tdfs.c	depth-first-search of a tree
cs_transpose.c	transpose a sparse matrix
cs_updown.c	sparse rank-1 Cholesky update/downate
cs_usolve.c	x=U\b
cs_util.c	various utilities (allocate/free matrices, workspace, etc)
cs_utsolve.c	x=U'\b
Makefile	Makefile for CSparse
README.txt	README file for CSparse


--------------------------------------------------------------------------------
./Demo:		C program demos
--------------------------------------------------------------------------------

cs_demo1.c	read a matrix from a file and perform basic matrix operations
cs_demo2.c	read a matrix from a file and solve a linear system
cs_demo3.c	read a matrix, solve a linear system, update/downdate
cs_demo.c	support routines for cs_demo*.c
cs_demo.h	include file for demo programs
cs_demo.out	output of "make", which runs the demos on some matrices
Makefile	Makefile for Demo programs
readhb.f	read a Rutherford-Boeing matrix
README.txt	Demo README file


--------------------------------------------------------------------------------
./MATLAB:	MATLAB interface, demos, and tests
--------------------------------------------------------------------------------

cs_install.m	MATLAB function for compiling and installing CSparse for MATLAB
CSparse/	MATLAB interface for CSparse
Demo/		MATLAB demos for CSparse
Makefile	MATLAB interface Makefile
README.txt	MATLAB README file
Test/		MATLAB test for CSparse, and "textbook" routines
UFget/		MATLAB interface to UF Sparse Matrix Collection


--------------------------------------------------------------------------------
./MATLAB/CSparse:   MATLAB interface for CSparse
--------------------------------------------------------------------------------

Contents.m	    Contents of MATLAB interface to CSparse
cs_add.m	    add two sparse matrices
cs_add_mex.c
cs_amd.m	    approximate minimum degree
cs_amd_mex.c
cs_chol.m	    sparse Cholesky
cs_chol_mex.c
cs_cholsol.m	    x=A\b using a sparse Cholesky
cs_cholsol_mex.c
cs_counts.m	    column counts for Cholesky or QR (like "symbfact" in MATLAB)
cs_counts_mex.c
cs_dmperm.m	    Dulmage-Mendelsohn permutation
cs_dmperm_mex.c
cs_dmsol.m	    x=A\b using dmperm
cs_dmspy.m	    plot a picture of a dmperm-permuted matrix
cs_droptol.m	    drop small entries
cs_droptol_mex.c
cs_esep.m	    find edge separator
cs_etree.m	    compute elimination tree
cs_etree_mex.c
cs_gaxpy.m	    sparse matrix times dense vector
cs_gaxpy_mex.c
cs_lsolve.m	    x=L\b where L is lower triangular
cs_lsolve_mex.c
cs_ltsolve.m	    x=L'\b where L is lower triangular
cs_ltsolve_mex.c
cs_lu.m		    sparse LU factorization
cs_lu_mex.c
cs_lusol.m	    x=A\b using sparse LU factorization
cs_lusol_mex.c
cs_make.m	    compiles CSparse for use in MATLAB
cs_mex.c	    support routines for CSparse mexFunctions
cs_mex.h
cs_multiply.m	    sparse matrix multiply
cs_multiply_mex.c
cs_must_compile.m   determine if a source file needs to be compiled with mex
cs_nd.m		    nested dissection
cs_nsep.m	    find node separator
cs_permute.m	    permute a sparse matrix
cs_permute_mex.c
cs_print.m	    print a sparse matrix
cs_print_mex.c
cs_qleft.m	    apply Householder vectors to the left
cs_qright.m	    apply Householder vectors to the right
cs_qr.m		    sparse QR factorization
cs_qr_mex.c
cs_qrsol.m	    solve a sparse least squares problem
cs_qrsol_mex.c
cs_randperm.m	    randdom permutation
cs_randperm_mex.c
cs_scc.m	    strongly-connected components
cs_scc_mex.c
cs_sep.m	    convert an edge separator into a node separator
cs_sparse.m	    convert a triplet form matrix to a compress-column form
cs_sparse_mex.c
cs_symperm.m	    symmetric permutation of a sparse matrix
cs_symperm_mex.c
cs_sqr.m	    symbolic QR ordering and analysis
cs_sqr_mex.c
cs_thumb_mex.c	    compute small "thumbnail" of a sparse matrix (for cspy).
cs_transpose.m	    transpose a sparse matrix
cs_transpose_mex.c
cs_updown.m	    sparse Cholesky update/downdate
cs_updown_mex.c
cs_usolve.m	    x=U\b where U is upper triangular 
cs_usolve_mex.c
cs_utsolve.m	    x=U'\b where U is upper triangular 
cs_utsolve_mex.c
cspy.m		    a color "spy"
Makefile	    Makefile for CSparse MATLAB interface
README.txt	    README file for CSparse MATLAB interface


--------------------------------------------------------------------------------
./MATLAB/Demo:	    MATLAB demos for CSparse
--------------------------------------------------------------------------------

Contents.m	    Contents of MATLAB demo for CSparse
cs_demo.m	    run all MATLAB demos for CSparse
cs_demo1.m	    MATLAB version of Demo/cs_demo1.c
cs_demo2.m	    MATLAB version of Demo/cs_demo2.c
cs_demo3.m	    MATLAB version of Demo/cs_demo3.c
private/	    private functions for MATLAB demos
README.txt	    README file for CSparse MATLAB demo


--------------------------------------------------------------------------------
./MATLAB/Demo/private: private functions for MATLAB demos
--------------------------------------------------------------------------------

demo2.m		    demo 2
demo3.m		    demo 3
ex1.m		    example 1
ex2.m		    example 2
ex3.m		    example 3
frand.m		    generate a random finite-element matrix
get_problem.m	    get a matrix
is_sym.m	    determine if a matrix is symmetric
mesh2d1.m	    construct a 2D mesh (method 1)
mesh2d2.m	    construct a 2D mesh (method 2)
mesh3d1.m	    construct a 3D mesh (method 1)
mesh3d2.m	    construct a 3D mesh (method 2)
print_order.m	    print the ordering method used
resid.m		    compute residual
rhs.m		    create right-hand-side


--------------------------------------------------------------------------------
./MATLAB/Test:	    Extensive test of CSparse, in MATLAB
--------------------------------------------------------------------------------

choldn.m	    Cholesky downdate
chol_downdate.m	    Cholesky downdate
chol_left2.m	    left-looking Cholesky
chol_left.m	    left-looking Cholesky
chol_right.m	    right-looking Cholesky
chol_super.m	    "supernodal" Cholesky
chol_update.m	    Cholesky update
chol_updown.m	    Cholesky update/downdate
cholupdown.m	    Cholesky update/downdate
chol_up.m	    up-looking Cholesky
cholup.m	    Cholesky update
cond1est.m	    1-norm condition estimate

Contents.m	    Contents of MATLAB/Test, "textbook" files only

cs_fiedler.m	    Fiedler vector
cs_frand_mex.c	    generate a random finite-element matrix
cs_ipvec_mex.c	    interface for cs_ipvec
cs_maxtransr_mex.c  recursive max transveral
cs_pvec_mex.c	    interface for cs_pvec
cs_q1.m		    construct Q from Householder vectors
cs_reachr_mex.c     recursive x=spones(L\sparse(b))
cs_reach_mex.c	    non-recursive x=spones(L\sparse(b))
cs_rowcnt_mex.c	    row counts for sparse Cholesky
cs_sparse2_mex.c    like cs_sparse, but for testing cs_entry
cs_test_make.m	    compiles MATLAB tests

dd.m		    dmperm test
e1.m		    etree test
ex.m		    an example
ff.m		    sparse QR example
givens2.m	    Givens rotation
gqr3.m		    Givens-based sparse QR
happly.m	    apply Householder
hh.m		    color test for cspy
hmake1.m	    construct a Householder reflection
hmake.m		    construct a Householder reflection
house.m		    Householder reflection
left_lu.m	    left-looking LU
lu_left.m	    left-looking LU
lu_right.m	    right-looking LU
lu_rightp.m	    right-looking LU, with partial pivoting
lu_rightpr.m	    recursive right-looking LU, with partial pivoting
lu_rightr.m	    recursive right-looking LU

Makefile	    Makefile for MATLAB Test directory

mynormest1.m	    1-norm estimate
myqr.m		    Householder QR
norm1est.m	    1-norm estimate
oo.m		    color test
pp.m		    cspy and cs_dmspy test
qr2.m		    Householder QR
qr_givens_full.m    Givens-based QR
qr_givens.m	    "sparse" Givens-based QR, using the etree
qr_left.m	    left-looking Householder QR
qr_right.m	    right-looking Householder QR

README.txt	    README file for MATLAB/Test

same.m		    determine if two vectors are the same
signum.m	    signum function

testall.m	    runs test1 to test28
test1.m		    gaxpy and triplet test
test2.m		    sparse triplet test
test3.m		    chol, x=L\b, x=L'\b, x=U\b, test
test4.m		    sparse matrix multiply test
test5.m		    sparse matrix add test
test6.m		    "textbook" (recursive) x=spones(L\sparse(b)) test
test7.m		    LU test
test8.m		    x=A\b test
test9.m		    QR test
test10.m	    QR test
test11.m	    row-count test
test12.m	    QR test
test13.m	    etree, counts test
test14.m	    droptol test
test15.m	    amd ordering test (requires MATLAB interface to AMD)
test16.m	    ordering test
test17.m	    QR test
test18.m	    iterative refinement test
test19.m	    dmperm, max transversal test
test20.m	    update/downdate test
test21.m	    update/downdate test
test22.m	    condition estimate test
test23.m	    dmspy test
test24.m	    Fiedler vector test
test25.m	    nested dissection test
test26.m	    dmsolve test
test27.m	    QR and utsolve test
test28.m	    randperm and dmperm test

testh.m		    Householder test
tg.m		    QR test
tp.m		    permutation tests
tqr2.m		    QR test
tqr.m		    QR test
tt.m		    cs_dmspy and graph separator tests


--------------------------------------------------------------------------------
./MATLAB/UFget:	    MATLAB interface for the UF Sparse Matrix Collection
--------------------------------------------------------------------------------

Contents.m	    Contents of UFget
mat/		    default directory where downloaded matrices will be put
README.txt	    README file for UFget
UFget_defaults.m    default parameter settings
UFget_example.m	    example of use
UFget_install.m	    installs UFget temporarily (for current session)
UFget_java.class    read a url and load it in into MATLAB (compiled Java code)
UFget_java.java	    read a url and load it in into MATLAB (Java source code)
UFget_lookup.m	    look up a matrix in the index
UFget.m		    UFget itself (primary user interface)
UFweb.m		    open url for a matrix or collection
mat/UF_Index.mat    index of matrices in UF Sparse Matrix Collection


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


--------------------------------------------------------------------------------
./Tcov:		    Exhaustive test coverage of CSparse
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
README.txt	    README file for Tcov directory

