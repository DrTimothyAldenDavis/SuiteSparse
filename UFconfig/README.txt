This file contains configuration settings for
all many of the software packages that I develop or
co-author:

  Package Version	    Description
  ------- -------	    -----------
  AMD	   1.2 or later	    approximate minimum degree ordering
  CAMD	   any
  COLAMD   2.4 or later	    column approximate minimum degree ordering
  CCOLAMD  any		    constrained approximate minimum degree ordering
  UMFPACK  4.5 or later	    sparse LU factorization, with the BLAS
  CXSparse any
  CHOLMOD  any		    sparse Cholesky factorization, update/downdate
  KLU	   1.0 or later	    sparse LU factorization, BLAS-free
  BTF	   1.0 or later	    permutation to block triangular form
  LDL	   any		    concise sparse LDL'
  LPDASA   any		    LP Dual Active Set Algorithm

In addition, the xerbla/ directory contains Fortan and C versions
of the BLAS/LAPACK xerbla routine, which is called when an invalid
input is passed to the BLAS or LAPACK.  The xerbla provided here
does not print any message, so the entire Fortran I/O library does
not need to be linked into a C application.  Most versions of the
BLAS contain xerbla, but those from K. Goto do not.  Use this if
you need too.
