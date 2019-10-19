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

All codes, below, are stable except KLU and BTF.  KLU and BTF are "beta", but
robust enough for production use.  They merely are still in development, and
are missing a few minor features to make them "1.0".  Thus the "beta".  They
are bug-free as far as I know, and are in use in commercial circuit simulation
packages.

Sept 11, 2006.  SuiteSparse version 2.1.1.  Note that SuiteSparse is now given its
own version number, rather than merely a date of release.  The Feb 6, 2006
version is "version 1.0.0".

UF suite of sparse matrix algorithms:

    AMD		approximate minimum degree ordering

    CAMD	constrained column approximate minimum degree ordering

    COLAMD	column approximate minimum degree ordering

    CCOLAMD	constrained column approximate minimum degree ordering

    BTF		permutation to block triangular form (beta)

    KLU		sparse LU factorization, primarily for circuit simulation.
		Requires AMD, COLAMD, and BTF (beta).  Optionally
		uses CHOLMOD, CCOLAMD, and METIS.

    UMFPACK	sparse LU factorization.  Requires AMD and the BLAS.

    CHOLMOD	sparse Cholesky factorization.  Requires AMD, COLAMD, CCOLAMD,
		METIS, the BLAS, and LAPACK.

    UFconfig	configuration file for all the above packages.  The
		UFconfig/UFconfig.mk is included in the Makefile's of all
		packages.  CSparse and CXSparse do not use UFconfig.

    CSparse	a concise sparse matrix package, developed for my upcoming
		book, "Direct Methods for Sparse Linear Systems", to be
		published by SIAM.

    CXSparse	CSparse Extended.  Includes support for complex matrices
		and both int or long integers.

    LPDASA	LP dual active set algorithm (to appear)

See http://www.netlib.org/blas for the Fortran reference BLAS (slow, but they
work).  See http://www.tacc.utexas.edu/~kgoto/ or
http://www.cs.utexas.edu/users/flame/goto/ for an optimized BLAS.
See http://www.netlib.org/lapack for LAPACK.  The UFconfig/UFconfig.mk
file assumes the Goto BLAS; change -lgoto to -l(your BLAS library here),
if you have another BLAS (-lblas, for example).

CHOLMOD requires METIS 4.0.1 (http://www-users.cs.umn.edu/~karypis/metis)
by default.  Place a copy of the metis-4.0 directory in the same directory
(SuiteSparse) containing this README file.  cd to metis-4.0 and type "make".
Edit the UFconfig/UFconfig.mk file (see that file for instructions), if
necessary.  Next, type "make" in this directory to compile all packages in
this distribution.  CHOLMOD can be compiled without METIS (use -DNPARTITION).

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis, CISE Dept., Univ. of
Florida.  email: my last name @ cise dot ufl dot edu.

To compile each package, cd to the top-level directory (AMD, COLAMD, etc)
and type "make".  Type "make clean" in the same directory to remove all but
the compiled libraries.  Type "make distclean" to remove all files not in
the original distribution.  Alternatively, just type "make" in this directory.

If you intend on compiling the MATLAB mexFunction interfaces, UFconfig.mk
should use

    CFLAGS = -O3 -fexceptions

(for Linux), to ensure that exceptions are properly caught.  See your
default MATLAB mexopts.sh file for how to do this for other systems
(type the command "mex -v").  Alternatively, you can use the various M-files
in each package to compile them from within the MATLAB Command Window, or
just type "SuiteSparse_install" in the MATLAB Command Window when MATLAB's
working directory is this one.  That is the only way to compile these packages
for Windows, unless you have Cygwin or wish to write your own MS Visual Studio
scripts.

--------------------------------------------------------------------------------

To turn on debugging (for development only, not needed by the typical user):

SuiteSparse/UFconfig/UFconfig.mk
    change CFLAGS = -O to CFLAGS = -g

To turn on debugging, add the line "#undef NDEBUG" in the following files.
To turn off debugging, remove that line.

    SuiteSparse/CHOLMOD/Include/cholmod_internal.h
    SuiteSparse/AMD/Source/amd_internal.h
    SuiteSparse/CAMD/Source/camd_internal.h
    SuiteSparse/CCOLAMD/ccolmod.c
    SuiteSparse/COLAMD/colamd.c
