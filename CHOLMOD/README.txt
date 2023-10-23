CHOLMOD: a sparse CHOLesky MODification package, Copyright (c) 2005-2023.
http://www.suitesparse.com
-----------------------------------------------

    CHOLMOD is a set of routines for factorizing sparse symmetric positive
    definite matrices of the form A or AA', updating/downdating a sparse
    Cholesky factorization, solving linear systems, updating/downdating
    the solution to the triangular system Lx=b, and many other sparse matrix
    functions for both symmetric and unsymmetric matrices.  Its supernodal
    Cholesky factorization relies on LAPACK and the Level-3 BLAS, and obtains
    a substantial fraction of the peak performance of the BLAS.  Both real and
    complex matrices are supported.  CHOLMOD is written in ANSI/ISO C, with both
    C and MATLAB interfaces.  This code works on Microsoft Windows and many
    versions of Unix and Linux.

CHOLMOD v5.0.0 introduces the first part of support for single precision
sparse matrices, with the introduction of the new CHOLMOD:Utility Module.
Single precision is not yet incorporated into the remaining Modules, however.

One CHOLMOD Module is copyrighted by the University of Florida (the
Partition Module).  The rest are copyrighted by the authors:
Timothy A. Davis (all of them), and William W. Hager (the Modify Module).

CHOLMOD relies on several other packages:  AMD, CAMD, COLAMD, CCOLAMD,
SuiteSparse_config, METIS, the BLAS, and LAPACK.  All but METIS, the BLAS, and
LAPACK are part of SuiteSparse.

AMD is authored by T. Davis, Iain Duff, and Patrick Amestoy.
COLAMD is authored by T. Davis and Stefan Larimore, with algorithmic design
in collaboration with John Gilbert and Esmond Ng.
CCOLAMD is authored by T. Davis and Siva Rajamanickam.
CAMD is authored by T. Davis and Y. Chen.

LAPACK and the BLAS are authored by Jack Dongarra and many others.
LAPACK is available at http://www.netlib.org/lapack

METIS 5.1.0 is authored by George Karypis, Univ. of Minnesota.  Its use in
CHOLMOD is optional.  A copy is in SuiteSparse/SuiteSparse_metis, and it is
slightly modified from the original METIS 5.1.0 to incorporate it into
SuiteSparse.

If you do not wish to use METIS, compile with -DNPARTITION.

For use in MATLAB (on any system, including Windows):  start MATLAB,
cd to the CHOLMOD/MATLAB directory, and type cholmod_make in the MATLAB
Command Window.

Acknowledgements:  this work was supported in part by the National Science
Foundation (NFS CCR-0203270 and DMS-9803599), and a grant from Sandia National
Laboratories (Dept. of Energy) which supported the development of CHOLMOD's
Partition Module.

