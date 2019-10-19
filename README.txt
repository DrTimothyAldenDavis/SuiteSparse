SuiteSparse:  A Suite of Sparse matrix packages

------------------
SuiteSparse/README
------------------

================================================================================
QUICK START FOR MATLAB USERS:  unzip the SuiteSparse.zip file, then in the
MATLAB Command Window, cd to the SuiteSparse directory and type
SuiteSparse_install.  All packages will be compiled, and several demos will be
run.
================================================================================

May 31, 2007.  SuiteSparse version 3.0.

UF suite of sparse matrix algorithms:

    AMD		approximate minimum degree ordering

    CAMD	constrained approximate minimum degree ordering

    COLAMD	column approximate minimum degree ordering

    CCOLAMD	constrained column approximate minimum degree ordering

    BTF		permutation to block triangular form

    KLU		sparse LU factorization, primarily for circuit simulation.
		Requires AMD, COLAMD, and BTF.  Optionally uses CHOLMOD,
		CAMD, CCOLAMD, and METIS.

    UMFPACK	sparse LU factorization.  Requires AMD and the BLAS.

    CHOLMOD	sparse Cholesky factorization.  Requires AMD, COLAMD, CCOLAMD,
		the BLAS, and LAPACK.  Optionally uses METIS.

    UFconfig	configuration file for all the above packages.  The
		UFconfig/UFconfig.mk is included in the Makefile's of all
		packages.  CSparse and RBio do not use UFconfig.

    CSparse	a concise sparse matrix package, developed for my upcoming
		book, "Direct Methods for Sparse Linear Systems", to be
		published by SIAM.

    CXSparse	CSparse Extended.  Includes support for complex matrices
		and both int or long integers.

    RBio	read/write sparse matrices in Rutherford/Boeing format

    UFcollection    toolbox for managing the UF Sparse Matrix Collection

    LPDASA	LP dual active set algorithm (to appear)


CHOLMOD and KLU optionally use METIS 4.0.1
(http://www-users.cs.umn.edu/~karypis/metis).  Place a copy of the metis-4.0
directory in the same directory (SuiteSparse) containing this README file.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis, CISE Dept., Univ. of
Florida.  email: my last name @ cise dot ufl dot edu.

================================================================================
If you use SuiteSparse_install in MATLAB, stop reading here.
================================================================================



----------------------------
To use "make" in Unix/Linux:
----------------------------

    See http://www.netlib.org/blas for the Fortran reference BLAS (slow, but
    they work).  See http://www.tacc.utexas.edu/~kgoto/ or
    http://www.cs.utexas.edu/users/flame/goto/ for an optimized BLAS.  See
    http://www.netlib.org/lapack for LAPACK.  The UFconfig/UFconfig.mk file
    assumes the vanilla BLAS (-lblas).  You should use an optimized BLAS;
    otherwise UMFPACK and CHOLMOD will be slow.  Change -lblas to -l(your BLAS
    library here).

    cd to metis-4.0 and edit the Makefile.in file.  I recommend making these
    changes to metis-4.0/Makefile.in:

    CC = gcc
    OPTFLAGS = -O3
    COPTIONS = -fexceptions -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE

    then type "make".  Now compile CHOLMOD.

    To compile all the C-callable libraries in SuiteSparse:  First, edit the
    UFconfig/UFconfig.mk file (see that file for instructions), if necessary.
    Next, type "make" in this directory to compile all packages in this
    distribution.  CHOLMOD can be compiled without METIS (use -DNPARTITION);
    this option is handled by SuiteSparse_install.m in MATLAB automatically, if
    the metis-4.0 directory does not appear (in the same directory as CHOLMOD,
    AMD, UMFPACK, etc).

    To compile each package, cd to the top-level directory (AMD, COLAMD, etc)
    and type "make".  Type "make clean" in the same directory to remove all but
    the compiled libraries.  Type "make distclean" to remove all files not in
    the original distribution.  Alternatively, just type "make" in this
    directory.

