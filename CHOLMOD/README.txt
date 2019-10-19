CHOLMOD: a sparse CHOLesky MODification package, Copyright (c) 2005-2014.
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


Some Modules of CHOLMOD are copyrighted by the University of Florida (the
Core and Partition Modules).  The rest are copyrighted by the authors:
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
CHOLMOD is optional.  A copy is in SuiteSparse/metis-5.1.0.

If you do not wish to use METIS, you must edit SuiteSparse_config and change
the line:

    CHOLMOD_CONFIG =

to

    CHOLMOD_CONFIG = -DNPARTITION

The CHOLMOD, AMD, COLAMD, CCOLAMD, and SuiteSparse)config directories must all
reside in a common parent directory.  To compile all these libraries, edit
SuiteSparse)config/SuiteSparse)config.mk to reflect your environment (C
compiler, location of the BLAS, and so on) and then type "make" in either the
CHOLMOD directory or in the parent directory of CHOLMOD.  See each package for
more details on how to compile them.

For use in MATLAB (on any system, including Windows):  start MATLAB,
cd to the CHOLMOD/MATLAB directory, and type cholmod_make in the MATLAB
Command Window.  This is the best way to compile CHOLMOD for MATLAB; it
provides a workaround for a METIS design feature, in which METIS terminates
your program (and thus MATLAB) if it runs out of memory.  Using cholmod_make
also ensures your mexFunctions are compiled with -fexceptions, so that
exceptions are handled properly (when hitting control-C in the MATLAB command
window, for example).

Acknowledgements:  this work was supported in part by the National Science
Foundation (NFS CCR-0203270 and DMS-9803599), and a grant from Sandia National
Laboratories (Dept. of Energy) which supported the development of CHOLMOD's
Partition Module.

