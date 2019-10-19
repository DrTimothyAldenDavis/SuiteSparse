SuiteSparse_config, 2014, Timothy A. Davis, http://www.suitesparse.com
(formerly the UFconfig package)

This directory contains a default SuiteSparse_config.mk file, which
in the original distribution is the same as SuiteSparse_config_linux.mk.
The various config file versions are:

    SuiteSparse_config_GPU_icc.mk   for GPU with the Intel compiler
    SuiteSparse_config_GPU_icc10.mk ditto, but for 10 cores
    SuiteSparse_config_GPU_gcc.mk   for GPU with the gcc compiler
    SuiteSparse_config_linux.mk     for linux, no GPU
    SuiteSparse_config_Mac.mk       for Mac
    SuiteSparse_config.mk           the actual one in use

To use a GPU for CHOLMOD, using gcc, do this:

    cp SuiteSparse_config_GPU_gcc.mk SuiteSparse_config.mk

To use a GPU for CHOLMOD, using icc and the Intel MKL, do this:

    cp SuiteSparse_config_GPU_icc.mk SuiteSparse_config.mk

To compile SuiteSparse for the Mac, do this:

    cp SuiteSparse_config_Mac.mk SuiteSparse_config.mk

To use a GPU for CHOLMOD, using icc and the Intel MKL,
and for a system with 10 cores, do this

    cp SuiteSparse_config_GPU_icc10.mk SuiteSparse_config.mk

For other alternatives, see the comments in the SuiteSparse_config.mk file.

--------------------------------------------------------------------------------

SuiteSparse_config contains configuration settings for all many of the software
packages that I develop or co-author.  Note that older versions of some of
these packages do not require SuiteSparse_config.

  Package  Description
  -------  -----------
  AMD      approximate minimum degree ordering
  CAMD     constrained AMD
  COLAMD   column approximate minimum degree ordering
  CCOLAMD  constrained approximate minimum degree ordering
  UMFPACK  sparse LU factorization, with the BLAS
  CXSparse int/long/real/complex version of CSparse
  CHOLMOD  sparse Cholesky factorization, update/downdate
  KLU      sparse LU factorization, BLAS-free
  BTF      permutation to block triangular form
  LDL      concise sparse LDL'
  LPDASA   LP Dual Active Set Algorithm
  RBio     read/write files in Rutherford/Boeing format
  SPQR     sparse QR factorization (full name: SuiteSparseQR)

SuiteSparse_config is not required by these packages:

  CSparse       a Concise Sparse matrix package
  MATLAB_Tools  toolboxes for use in MATLAB

In addition, the xerbla/ directory contains Fortan and C versions of the
BLAS/LAPACK xerbla routine, which is called when an invalid input is passed to
the BLAS or LAPACK.  The xerbla provided here does not print any message, so
the entire Fortran I/O library does not need to be linked into a C application.
Most versions of the BLAS contain xerbla, but those from K. Goto do not.  Use
this if you need too.

If you edit this directory (SuiteSparse_config.mk in particular) then you
must do "make purge ; make" in the parent directory to recompile all of
SuiteSparse.  Otherwise, the changes will not necessarily be applied.

