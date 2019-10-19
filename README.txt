SuiteSparse:  A Suite of Sparse matrix packages

------------------
SuiteSparse/README
------------------

================================================================================
QUICK START FOR MATLAB USERS:  uncompress the SuiteSparse.zip or
SuiteSparse.tar.gz archive file (they contain the same thing), then in the
MATLAB Command Window, cd to the SuiteSparse directory and type
SuiteSparse_install.  All packages will be compiled, and several demos will be
run.
================================================================================

May 20, 2009.  SuiteSparse version 3.4.0

    AMD         approximate minimum degree ordering

    CAMD        constrained approximate minimum degree ordering

    COLAMD      column approximate minimum degree ordering

    CCOLAMD     constrained column approximate minimum degree ordering

    BTF         permutation to block triangular form

    KLU         sparse LU factorization, primarily for circuit simulation.
                Requires AMD, COLAMD, and BTF.  Optionally uses CHOLMOD,
                CAMD, CCOLAMD, and METIS.

    UMFPACK     sparse LU factorization.  Requires AMD and the BLAS.

    CHOLMOD     sparse Cholesky factorization.  Requires AMD, COLAMD, CCOLAMD,
                the BLAS, and LAPACK.  Optionally uses METIS.

    UFconfig    configuration file for all the above packages.  The
                UFconfig/UFconfig.mk is included in the Makefile's of all
                packages.  CSparse and RBio do not use UFconfig.

    CSparse     a concise sparse matrix package, developed for my upcoming
                book, "Direct Methods for Sparse Linear Systems", to be
                published by SIAM.

    CXSparse    CSparse Extended.  Includes support for complex matrices
                and both int or long integers.

    RBio        read/write sparse matrices in Rutherford/Boeing format

    UFcollection    toolbox for managing the UF Sparse Matrix Collection

    LPDASA      LP dual active set algorithm (to appear)

    MESHND      2D and 3D mesh generation and nested dissection ordering

    SSMULT      sparse matrix multiply for MATLAB

    LINFACTOR   simple m-file demonstrating how to use LU and CHOL in
                MATLAB to solve Ax=b

    MATLAB_Tools    various simple m-files for use in MATLAB

    SuiteSparseQR   sparse QR factorization

CHOLMOD optionally uses METIS 4.0.1
(http://www-users.cs.umn.edu/~karypis/metis).  To use METIS, place a copy of
the metis-4.0 directory in the same directory containing this README file.
Be sure that you do not have a nested metis-4.0/metis-4.0 directory; SuiteSparse
won't find METIS if you do this, which can happen with a zip file of metis-4.0
on Windows.  The use of METIS will improve the ordering quality in CHOLMOD.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis, CISE Dept., Univ. of
Florida.  email: my last name @ cise dot ufl dot edu.

================================================================================
If you use SuiteSparse_install in MATLAB, stop reading here.
================================================================================



----------------------------
To use "make" in Unix/Linux:
----------------------------

(1) Use the right BLAS and LAPACK libraries

    See http://www.netlib.org/blas for the Fortran reference BLAS (slow, but
    they work).  See http://www.tacc.utexas.edu/~kgoto/ or
    http://www.cs.utexas.edu/users/flame/goto/ for an optimized BLAS.  See
    http://www.netlib.org/lapack for LAPACK.  The UFconfig/UFconfig.mk file
    assumes the vanilla BLAS (-lblas).  You should use an optimized BLAS;
    otherwise UMFPACK and CHOLMOD will be slow.  Change -lblas to -l(your BLAS
    library here) in the UFconfig/UFconfig.mk file.

(2) Install Intel's Threading Building Blocks (TBB)

    This is optionally used by SuiteSparseQR.  Refer to the User Guide in 
    SuiteSparse/SPQR/Doc/spqr_user_guide.pdf for details.

(3) Configure METIS (or don't use METIS)

    cd to metis-4.0 and edit the Makefile.in file.  I recommend making these
    changes to metis-4.0/Makefile.in:

        CC = gcc
        OPTFLAGS = -O3

    And, if you want to use METIS in MATLAB and compile with "make" instead
    of using SuiteSparse_install.m:

        COPTIONS = -fexceptions -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE

    Next, cd to metis-4.0 and type "make".

    If you do not wish to use METIS, then edit the UFconfig/UFconfig.mk file,
    and change the lines

        CHOLMOD_CONFIG =
        SPQR_CONFIG =

    to

        CHOLMOD_CONFIG = -DNPARTITION
        SPQR_CONFIG = -DNPARTITION

    Also change the line

        METIS = ../../metis-4.0/libmetis.a

    to

        METIS =

(4) Make other changes to UFconfig/UFconfig.mk as needed

    Edit the UFconfig/UFconfig.mk file as needed.  Directions are in that file.
    If you have compiled SuiteSparse already (partially or completely), then
    whenever you edit the UFconfig/UFconfig.mk file, you should then type
    "make purge" (or "make realclean") in this directory.

    If you want to use "make" to compile mexFunctions, I recommend adding
    these options to the CFLAGS = line:

        -fexceptions -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE

(5) Type "make" in this directory.  All packages will be be compiled.  METIS
    will be compiled if you have it.  Several demos will be run.

    The libraries will appear in */Lib/*.a.  Include files, as needed by user
    programs that use CHOLMOD, AMD, CAMD, COLAMD, CCOLAMD, BTF, KLU, UMFPACK,
    LDL, etc. are in */Include/*.h.

    The METIS library is in metis-4.0/libmetis.a.  METIS Include files (not
    needed by the end user of SuiteSparse) are in located in metis-4.0/Lib/*.h.


In a future version, I will include a "make install" that will create *.so
libraries and place them in /usr/lib.  The libraries should be called
libPACKAGE.so.VERSION.SUBVERSION.SUBSUBVERSION.  For example, 
libcolamd.so.2.7.1 should be the library name for COLAMD version 2.7.1.
The version numbers are located in UFconfig.h (in comments) and in each
package (as a #define).
