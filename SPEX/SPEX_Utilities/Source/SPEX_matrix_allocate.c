//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_matrix_allocate: allocate a SPEX_matrix
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Allocate an m-by-n SPEX_matrix, in either dynamic_CSC x mpz or one of
// 15 data structures:
// (sparse CSC, sparse triplet, or dense) x
// (mpz, mpq, mfpr, int64, or double).

// If the matrix is not dynamic_CSC, then it may be created as 'shallow', in
// which case A->p, A->i, A->j, and A->x are all returned as NULL, and all
// A->*_shallow flags are returned as true.

// If the matrix is dynamic_CSC, each column of the returned matrix will be
// allocated as SPEX_vector with zero available entry. Additional reallocation
// for each column will be needed.

#define SPEX_FREE_ALL                       \
{                                           \
    SPEX_matrix_free (&A, option);          \
}

#include "spex_util_internal.h"

#if defined (__GNUC__)
    #if ( __GNUC__ == 11)
        // gcc 11 has a bug that triggers a spurious warning for the call
        // to SPEX_MPQ_INIT (A->scale), from -Wstringop-overflow.  see
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101854
        #pragma GCC diagnostic ignored "-Wstringop-overflow"
    #endif
#endif

SPEX_info SPEX_matrix_allocate
(
    SPEX_matrix *A_handle,  // matrix to allocate
    SPEX_kind kind,         // CSC, triplet, dense, dynamic_CSC
    SPEX_type type,         // mpz, mpq, mpfr, int64, or double
    int64_t m,              // # of rows
    int64_t n,              // # of columns
    int64_t nzmax,          // max # of entries for CSC or triplet
                            // (ignored if A is dense)
    bool shallow,           // if true, matrix is shallow.  A->p, A->i, A->j,
                            // A->x are all returned as NULL and must be set
                            // by the caller.  All A->*_shallow are returned
                            // as true. Ignored if kind is dynamic_CSC.
    bool init,              // If true, and the data types are mpz, mpq, or
                            // mpfr, the entries are initialized (using the
                            // appropriate SPEX_mp*_init function). If false,
                            // the mpz, mpq, and mpfr arrays are malloced but
                            // not initialized. Utilized internally to reduce
                            // memory.  Ignored if shallow is true.
    const SPEX_options option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    if (!spex_initialized ( )) return (SPEX_PANIC);

    if (A_handle == NULL)
    {
        return (SPEX_INCORRECT_INPUT);
    }
    (*A_handle) = NULL ;
    if (m < 0 || n < 0 ||
        kind  < SPEX_CSC || kind  > SPEX_DENSE ||
        type  < SPEX_MPZ || type  > SPEX_FP64) 
    {
        return (SPEX_INCORRECT_INPUT);
    }

    //--------------------------------------------------------------------------
    // allocate the header
    //--------------------------------------------------------------------------

    SPEX_matrix A = (SPEX_matrix) SPEX_calloc (1, sizeof (SPEX_matrix_struct));
    if (A == NULL)
    {
        return (SPEX_OUT_OF_MEMORY);
    }

    if (kind == SPEX_DENSE)
    {
        nzmax = m*n ;
    }
    nzmax = SPEX_MAX (nzmax, 1);

    A->m = m ;
    A->n = n ;
    A->nzmax = nzmax ;
    A->nz = 0 ;             // for triplet matrices only (no triplets yet)
    A->kind = kind ;
    A->type = type ;

    // A->p, A->i, A->j, and A->x are currently NULL since A was calloc'd.
    A->p_shallow = false ;
    A->i_shallow = false ;
    A->j_shallow = false ;
    A->x_shallow = false ;

    // A->scale = 1
    SPEX_MPQ_INIT (A->scale) ;
    SPEX_MPQ_SET_UI (A->scale, 1, 1);

    //--------------------------------------------------------------------------
    // allocate the p, i, j, and x components
    //--------------------------------------------------------------------------

    if(shallow)
    {

        // all components are shallow.  The caller can modify individual
        // components after A is created, as needed.
        A->p_shallow = true ;
        A->i_shallow = true ;
        A->j_shallow = true ;
        A->x_shallow = true ;

    }
    else
    {

        bool ok = true ;

        // allocate the integer pattern
        switch (kind)
        {
            case SPEX_CSC:
                A->p = (int64_t *) SPEX_calloc (n+1, sizeof (int64_t));
                A->i = (int64_t *) SPEX_calloc (nzmax, sizeof (int64_t));
                ok = (A->p != NULL && A->i != NULL);
                break ;

            case SPEX_TRIPLET:
                A->i = (int64_t *) SPEX_calloc (nzmax, sizeof (int64_t));
                A->j = (int64_t *) SPEX_calloc (nzmax, sizeof (int64_t));
                ok = (A->i != NULL && A->j != NULL);
                break ;

            default: //SPEX_DENSE or SPEX_DYNAMIC_CSC
                ;// nothing to do

        }

        // allocate the values
        switch (type)
        {
            case SPEX_MPZ:
                // If init == true, we create and initialize each entry
                // in the integer array. If init == false, then we only
                // allocate the array but do not allocate the individual
                // mpz, mpq, or mpfr
                if (init)
                {
                    A->x.mpz = spex_create_mpz_array (nzmax);
                }
                else
                {
                    A->x.mpz = SPEX_calloc(nzmax, sizeof(mpz_t));
                }
                ok = ok && (A->x.mpz != NULL);
                break ;

            case SPEX_MPQ:
                if (init)
                {
                    A->x.mpq = spex_create_mpq_array (nzmax);
                }
                else
                {
                    A->x.mpq = SPEX_calloc(nzmax, sizeof(mpq_t));
                }
                ok = ok && (A->x.mpq != NULL);
                break ;

            case SPEX_MPFR:
                if (init)
                {
                    A->x.mpfr = spex_create_mpfr_array (nzmax, option);
                }
                else
                {
                    A->x.mpfr = SPEX_calloc(nzmax, sizeof(mpfr_t));
                }
                ok = ok && (A->x.mpfr != NULL);
                break ;

            case SPEX_INT64:
                A->x.int64 = (int64_t *) SPEX_calloc (nzmax, sizeof (int64_t));
                ok = ok && (A->x.int64 != NULL);
                break ;

            case SPEX_FP64:
                A->x.fp64 = (double *) SPEX_calloc (nzmax, sizeof (double));
                ok = ok && (A->x.fp64 != NULL);
                break ;

        }

        if (!ok)
        {
            SPEX_FREE_ALL;
            return (SPEX_OUT_OF_MEMORY);
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*A_handle) = A ;
    return (SPEX_OK);
}

