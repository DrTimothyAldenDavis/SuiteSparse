SuiteSparse:  A Suite of Sparse matrix packages at http://www.suitesparse.com

------------------
SuiteSparse/README
------------------

================================================================================
QUICK START FOR MATLAB USERS (Linux, Mac, or Windows):  uncompress the
SuiteSparse.zip or SuiteSparse.tar.gz archive file (they contain the same
thing), then in the MATLAB Command Window, cd to the SuiteSparse directory and
type SuiteSparse_install.  All packages will be compiled, and several demos
will be run.

QUICK START FOR LINUX:  Just type 'make' in this directory.  Then do
'sudo make install' if you want to install the libraries and include files
in /usr/local.

QUICK START FOR MAC:  Delete the SuiteSparse_config/SuiteSparse_config.mk
file, and then remove "_Mac" from the *Mac.mk file in that directory.  Then
continue as the 'QUICK START FOR LINUX' above.
================================================================================

July 17, 2012.  SuiteSparse VERSION 4.0.2

    spqr_rank   MATLAB toolbox for rank deficient sparse matrices: null spaces,
                reliable factorizations, etc.  With Leslie Foster, San Jose
                State Univ.

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

    SuiteSparse_config    configuration file for all the above packages.  The
                SuiteSparse_config/SuiteSparse_config.mk is included in the
                Makefile's of all packages.  CSparse and MATLAB_Tools do not
                use SuiteSparse_config.  Prior to SuiteSparse Version 4.0.0,
                this configuration directory was called 'UFconfig'.
                Version 4.0.0 and later use SuiteSparse_config instead,
                which is upward compatible with UFconfig.

    CSparse     a concise sparse matrix package, developed for my
                book, "Direct Methods for Sparse Linear Systems",
                published by SIAM.  Intended primarily for teaching.

    CXSparse    CSparse Extended.  Includes support for complex matrices
                and both int or long integers.  Use this instead of CSparse
                for production use.

    RBio        read/write sparse matrices in Rutherford/Boeing format

    UFcollection    toolbox for managing the UF Sparse Matrix Collection

    LPDASA      LP dual active set algorithm (to appear)

    MESHND      2D and 3D mesh generation and nested dissection ordering

    SSMULT      sparse matrix multiply for MATLAB

    LINFACTOR   simple m-file demonstrating how to use LU and CHOL in
                MATLAB to solve Ax=b

    MATLAB_Tools    various simple m-files for use in MATLAB

    SuiteSparseQR   sparse QR factorization

Some codes optionally use METIS 4.0.1
(http://www-users.cs.umn.edu/~karypis/metis).  To use METIS, place a copy of
the metis-4.0 directory in the same directory containing this README file.
Be sure that you do not have a nested metis-4.0/metis-4.0 directory; SuiteSparse
won't find METIS if you do this, which can happen with a zip file of metis-4.0
on Windows.  The use of METIS will improve the ordering quality.

Refer to each package for license, copyright, and author information.  All
codes are authored or co-authored by Timothy A. Davis.
email: DrTimothyAldenDavis@gmail.com

================================================================================
If you use SuiteSparse_install in MATLAB, stop reading here.
================================================================================



----------------------------
To use "make" in Unix/Linux:
----------------------------

(1) Use the right BLAS and LAPACK libraries

    Edit your SuiteSparse_config/SuiteSparse_config.mk file to point to the
    right compilers, and to the correct BLAS and LAPACK libraries.  There are
    many examples of different computer architectures there.  Scroll through to
    find yours, and uncomment those lines.

(2) Install Intel's Threading Building Blocks (TBB)

    This is optionally used by SuiteSparseQR.  Refer to the User Guide in 
    SuiteSparse/SPQR/Doc/spqr_user_guide.pdf for details.

(3) Configure METIS (or don't use METIS)

    If you don't download METIS, skip this step.  Otherwise,
    cd to metis-4.0 and edit the Makefile.in file.  I recommend making these
    changes to metis-4.0/Makefile.in, but this is optional.

        CC = gcc
        OPTFLAGS = -O3

(4) Make other changes to SuiteSparse_config/SuiteSparse_config.mk as needed

    Edit the SuiteSparse_config/SuiteSparse_config.mk file as needed.
    Directions are in that file.  If you have compiled SuiteSparse already
    (partially or completely), then whenever you edit the
    SuiteSparse_config/SuiteSparse_config.mk file, you should then type "make
    purge" (or "make realclean") in this directory.

(5) Type "make" in this directory.  All packages will be be compiled.  METIS
    will be compiled if you have it.  Several demos will be run.

    To compile just the libraries, without running any demos, use
    "make library".

    The libraries will appear in */Lib/*.a.  Include files, as needed by user
    programs that use CHOLMOD, AMD, CAMD, COLAMD, CCOLAMD, BTF, KLU, UMFPACK,
    LDL, etc. are in */Include/*.h.

    The METIS library is in metis-4.0/libmetis.a.  METIS Include files (not
    needed by the end user of SuiteSparse) are in located in metis-4.0/Lib/*.h.

(6) To install, type "sudo make install".  This will place copies of all
    libraries in /usr/local/lib, and all include files in /usr/local/include.
    You can change the install location by editting SuiteSparse_config.mk.
    These directories must already exist.

(7) To uninstall, type "sudo make uninstall", which reverses "make install"
    by removing the SuiteSparse libraries from /usr/local/lib, and the
    include files from /usr/local/include.
