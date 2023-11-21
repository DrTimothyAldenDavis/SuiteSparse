//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_cmread: test program that reads in a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Read in a matrix from a file and print it out.
//
// Usage:
//      cmread matrixfile
//      cmread < matrixfile
//
// With the option -x, the matrix is assumed to be mangled.

#include "cholmod.h"

#ifdef CHOLMOD_INT64
#define CHOLMOD(routine) cholmod_l_ ## routine
#define Int int64_t
#define UInt uint64_t
#else
#define CHOLMOD(routine) cholmod_ ## routine
#define Int int32_t
#define UInt uint32_t
#endif

#ifdef SINGLE
#define Real float
#define DTYPE CHOLMOD_SINGLE
#else
#define Real double
#define DTYPE CHOLMOD_DOUBLE
#endif

#define OK(expr)                            \
{                                           \
    if (!(expr))                            \
    {                                       \
        fprintf (stderr, "line %d Test FAIL\n", __LINE__) ;   \
        printf ("line %d Test FAIL\n", __LINE__) ;            \
        fflush (stdout) ;                   \
        fflush (stderr) ;                   \
        abort ( ) ;                         \
    }                                       \
}

#include "t_znorm_diag.c"

//------------------------------------------------------------------------------
// test_sparse
//------------------------------------------------------------------------------

void test_sparse (cholmod_sparse *C, cholmod_common *cm) ;

void test_sparse (cholmod_sparse *C, cholmod_common *cm)
{

    //--------------------------------------------------------------------------
    // print C and create Z
    //--------------------------------------------------------------------------

    CHOLMOD(print_sparse) (C, "C", cm) ;
    cholmod_sparse *Z = CHOLMOD(speye) (C->nrow, C->ncol,
        CHOLMOD_PATTERN + DTYPE, cm) ;
    Z->stype = C->stype ;
    CHOLMOD(print_sparse) (Z, "Z", cm) ;

    //--------------------------------------------------------------------------
    // symmetry
    //--------------------------------------------------------------------------

    for (int option = 0 ; option <= 2 ; option++)
    {
        Int xmatch = 0, pmatch = 0, nzoff = 0, nzd = 0 ;
        int asym = CHOLMOD(symmetry) (C, option, &xmatch, &pmatch, &nzoff,
            &nzd, cm) ;
        printf ("option: %d asym %d\n", option, asym) ;
    }

    //--------------------------------------------------------------------------
    // write C as a sparse matrix and read it back in as A
    //--------------------------------------------------------------------------

    FILE *f = fopen ("temp5.mtx", "w") ;
    CHOLMOD(write_sparse) (f, C, Z, NULL, cm) ;
    fclose (f) ;
    f = fopen ("temp5.mtx", "r") ;
    cholmod_sparse *A = CHOLMOD(read_sparse2) (f, DTYPE, cm) ;
    fclose (f) ;
    double anorm = CHOLMOD(norm_sparse) (A, 0, cm) ;
    double dnorm = znorm_diag (A, cm) ;

    //--------------------------------------------------------------------------
    // test C and A
    //--------------------------------------------------------------------------

    // do the test only if norm(A) is finite, and if norm(imag(diag(A))) is
    // zero or the matrix is unsymmetric.

    if (isfinite (anorm) && (dnorm == 0 || A->stype == 0))
    {

        //----------------------------------------------------------------------
        // compare C and A
        //----------------------------------------------------------------------

        double one [2] = {1,0} ;
        double minusone [2] = {-1,0} ;
        CHOLMOD(print_sparse) (C, "C", cm) ;
        CHOLMOD(print_sparse) (A, "A", cm) ;
        cholmod_sparse *E = CHOLMOD(add) (C, A, one, minusone, 2, true, cm) ;
        CHOLMOD(print_sparse) (E, "E", cm) ;
        double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        printf ("enorm %g\n", enorm) ;
        OK (enorm == 0) ;
        CHOLMOD(free_sparse) (&E, cm) ;

        //----------------------------------------------------------------------
        // test the pattern of C and A
        //----------------------------------------------------------------------

        CHOLMOD(sparse_xtype) (CHOLMOD_PATTERN, A, cm);
        CHOLMOD(sparse_xtype) (CHOLMOD_REAL, A, cm) ;
        cholmod_sparse *G = CHOLMOD(add) (C, Z, one, one, 2, false, cm);
        CHOLMOD(sparse_xtype) (CHOLMOD_REAL, G, cm) ;
        CHOLMOD(print_sparse) (G, "G", cm) ;
        CHOLMOD(print_sparse) (A, "A", cm) ;
        E = CHOLMOD(add) (G, A, one, minusone, 2, true, cm) ;
        CHOLMOD(print_sparse) (E, "E", cm) ;
        enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        printf ("pattern enorm %g\n", enorm) ;
        OK (enorm == 0) ;
        CHOLMOD(free_sparse) (&E, cm) ;
        CHOLMOD(free_sparse) (&G, cm) ;

        //----------------------------------------------------------------------
        // write C as a dense matrix and read it back in
        //----------------------------------------------------------------------

        cholmod_dense *X = CHOLMOD(sparse_to_dense) (C, cm) ;
        f = fopen ("temp6.mtx", "w") ;
        CHOLMOD(write_dense) (f, X, NULL, cm) ;
        fclose (f) ;
        f = fopen ("temp6.mtx", "r") ;
        cholmod_dense *Y = CHOLMOD(read_dense2) (f, DTYPE, cm) ;
        cholmod_sparse *X2 = CHOLMOD(dense_to_sparse) (X, true, cm) ;
        cholmod_sparse *Y2 = CHOLMOD(dense_to_sparse) (Y, true, cm) ;

        CHOLMOD(print_sparse) (X2, "X2", cm) ;
        CHOLMOD(print_sparse) (Y2, "Y2", cm) ;
        E = CHOLMOD(add) (X2, Y2, one, minusone, 2, true, cm) ;
        CHOLMOD(print_sparse) (E, "E", cm) ;
        enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        printf ("dense enorm %g\n", enorm) ;
        OK (enorm == 0) ;
        CHOLMOD(free_sparse) (&E, cm) ;

        CHOLMOD(free_sparse) (&X2, cm) ;
        CHOLMOD(free_sparse) (&Y2, cm) ;
        CHOLMOD(free_dense) (&Y, cm) ;
        CHOLMOD(free_dense) (&X, cm) ;
    }

    CHOLMOD(free_sparse) (&A, cm) ;
    CHOLMOD(free_sparse) (&Z, cm) ;
}

//------------------------------------------------------------------------------
// *_test: test read/write methods
//------------------------------------------------------------------------------

int main (int argc, char **argv)
{
    cholmod_sparse *C ;
    cholmod_dense *X ;
    cholmod_triplet *T ;
    void *V ;
    FILE *f ;
    cholmod_common Common, *cm ;
    int mtype, prefer ;

    //--------------------------------------------------------------------------
    // get the file containing the input matrix
    //--------------------------------------------------------------------------

    char *filename = NULL;
    bool mangled = false ;
    if (argc > 1)
    {
        char *arg = argv [1] ;
        if (arg [0] == '-' && arg [1] == 'x')
        {
            mangled = true ;
            filename = (argc > 2) ? argv [2] : NULL ;
        }
        else
        {
            filename = argv [1] ;
        }
    }

    if (filename == NULL)
    {
        f = stdin ;
    }
    else
    {
        if ((f = fopen (filename, "r")) == NULL)
        {
            printf ("cannot open file: %s\n", filename) ;
            return (0) ;
        }
    }

    //--------------------------------------------------------------------------
    // start CHOLMOD, read the matrix, print it, and free it
    //--------------------------------------------------------------------------

    cm = &Common ;
    CHOLMOD(start) (cm) ;
    cm->print = 5 ;

    //--------------------------------------------------------------------------
    // read the matrix (assuming it's sparse), print it, and free it
    //--------------------------------------------------------------------------

    cholmod_sparse *A = CHOLMOD(read_sparse2) (f, DTYPE, cm) ;
    if (argc > 1) fclose (f) ;
    if (A == NULL)
    {
        printf ("Matrix is mangled, or not sparse\n") ;
    }
    CHOLMOD(print_sparse) (A, "A", cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;

    //--------------------------------------------------------------------------
    // read the matrix in different formats
    //--------------------------------------------------------------------------

    if (argc > 1)
    {
        for (prefer = 0 ; prefer <= 2 ; prefer++)
        {
            printf ("\n---------------------- Prefer: %d\n", prefer) ;
            f = fopen (filename, "r") ;
            V = CHOLMOD(read_matrix2) (f, prefer, DTYPE, &mtype, cm) ;
            if (V == NULL)
            {
                printf ("Matrix is mangled\n") ;
                OK (mangled) ;
            }
            else
            {
                printf ("Matrix is OK\n") ;
                OK (!mangled) ;
                switch (mtype)
                {
                    case CHOLMOD_TRIPLET:
                        printf ("\n=========================TRIPLET:\n") ;
                        T = V ;
                        CHOLMOD(print_triplet) (T, "T", cm) ;
                        OK (T->dtype == DTYPE) ;
                        C = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
                        test_sparse (C, cm) ;
                        CHOLMOD(free_sparse) (&C, cm) ;
                        CHOLMOD(free_triplet) (&T, cm) ;
                        break ;

                    case CHOLMOD_SPARSE:
                        printf ("\n=========================SPARSE:\n") ;
                        C = V ;
                        CHOLMOD(print_sparse) (C, "C", cm) ;
                        OK (C->dtype == DTYPE) ;
                        test_sparse (C, cm) ;
                        CHOLMOD(free_sparse) (&C, cm) ;
                        break ;

                    case CHOLMOD_DENSE:
                        printf ("\n=========================DENSE:\n") ;
                        X = V ;
                        CHOLMOD(print_dense) (X, "X", cm) ;
                        OK (X->dtype == DTYPE) ;
                        C = CHOLMOD(dense_to_sparse) (X, true, cm) ;
                        test_sparse (C, cm) ;
                        CHOLMOD(free_sparse) (&C, cm) ;
                        CHOLMOD(free_dense) (&X, cm) ;
                        break ;
                }
            }
            fclose (f) ;
        }
    }

    CHOLMOD(finish) (cm) ;
    return (0) ;
}

