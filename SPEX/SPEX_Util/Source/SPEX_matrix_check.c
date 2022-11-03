//------------------------------------------------------------------------------
// SPEX_Util/SPEX_matrix_check: check if a matrix is OK
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_ALL    \
    SPEX_FREE (work) ;

#include "spex_util_internal.h"

// if pr == 2, turn off printing after 30 lines of output
#define SPEX_PR_LIMIT                       \
    lines++ ;                               \
    if (pr == 2 && lines > 30)              \
    {                                       \
        SPEX_PRINTF ("    ...\n") ;         \
        pr = 1 ;                            \
    }

int compar (const void *x, const void *y) ;

int compar (const void *x, const void *y)
{
    // compare two (i,j) tuples
    int64_t *a = (int64_t *) x ;
    int64_t *b = (int64_t *) y ;
    if (a [0] < b [0])
    {
        return (-1) ;
    }
    else if (a [0] > b [0])
    {
        return (1) ;
    }
    else if (a [1] < b [1])
    {
        return (-1) ;
    }
    else if (a [1] > b [1])
    {
        return (1) ;
    }
    else
    {
        return (0) ;
    }
}

/* check the validity of a SPEX_matrix */

// print_level from option struct:
//      0: nothing
//      1: just errors
//      2: errors and terse output
//      3: verbose

SPEX_info SPEX_matrix_check     // returns a SPEX status code
(
    const SPEX_matrix *A,     // matrix to check
    const SPEX_options* option
)
{

    if (!spex_initialized ( )) return (SPEX_PANIC) ;

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    SPEX_info status = 0 ;
    char * buff = NULL ;
    int pr = SPEX_OPTION_PRINT_LEVEL (option) ;
    int64_t nz;                            // Number of nonzeros in A
    status = SPEX_matrix_nnz(&nz, A, option);
    if (status != SPEX_OK) {return status;}

    int64_t m = A->m ;
    int64_t n = A->n ;
    int64_t nzmax = A->nzmax ;

    if (m < 0)
    {
        SPEX_PR1 ("m invalid\n") ;
        return (SPEX_INCORRECT_INPUT) ;
    }
    if (n < 0)
    {
        SPEX_PR1 ("n invalid\n") ;
        return (SPEX_INCORRECT_INPUT) ;
    }
    if (nzmax < 0)
    {
        SPEX_PR1 ("nzmax invalid\n") ;
        return (SPEX_INCORRECT_INPUT) ;
    }

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    if (A->type < SPEX_MPZ || A->type > SPEX_FP64)
    //  A->kind < SPEX_CSC || A->kind > SPEX_DENSE // checked in SPEX_matrix_nnz
    {
        SPEX_PR1 ("A has invalid type.\n") ;
        return (SPEX_INCORRECT_INPUT) ;
    }

    SPEX_PR2 ("SPEX_matrix: nrows: %"PRId64", ncols: %"PRId64", nz:"
        "%"PRId64", nzmax: %"PRId64", kind: %s, type: %s\n", m, n, nz,
        nzmax, A->kind < 1 ? "CSC" : A->kind < 2 ? "Triplet" : "Dense",
        A->type < 1 ? "MPZ" : A->type < 2 ? "MPQ" : A->type < 3 ?
        "MPFR" : A->type < 4 ? "int64" : "double") ;

    if (pr >= 2)
    {
        SPEX_PR2 ("scale factor: ") ;
        status = SPEX_mpfr_asprintf (&buff,"%Qd\n", A->scale) ;
        if (status >= 0)
        {
            SPEX_PR2 ("%s", buff) ;
            SPEX_mpfr_free_str (buff) ;
        }
    }

    //--------------------------------------------------------------------------
    // initialize workspace
    //--------------------------------------------------------------------------

    int64_t i, j, p, pend ;
    int64_t* work = NULL;   // used for checking duplicates for CSC and triplet
    uint64_t prec = SPEX_OPTION_PREC (option);

    int64_t lines = 0 ;     // # of lines printed so far

    //--------------------------------------------------------------------------
    // check each kind of matrix: CSC, triplet, or dense
    //--------------------------------------------------------------------------

    switch (A->kind)
    {

        //----------------------------------------------------------------------
        // check a matrix in CSC format
        //----------------------------------------------------------------------

        case SPEX_CSC:
        {
            int64_t* Ap = A->p;
            int64_t* Ai = A->i;

            //------------------------------------------------------------------
            // check the column pointers
            //------------------------------------------------------------------

            if (nzmax > 0 && (Ap == NULL || Ap [0] != 0))
            {
                // column pointers invalid
                SPEX_PR1 ("p invalid\n") ;
                return (SPEX_INCORRECT_INPUT) ;
            }
            for (j = 0 ; j < n ; j++)
            {
                p = Ap [j] ;
                pend = Ap [j+1] ;
                if (pend < p || pend > nz)
                {
                    // column pointers not monotonically non-decreasing
                    SPEX_PR1 ("p invalid\n") ;
                    return (SPEX_INCORRECT_INPUT) ;
                }
            }

            //------------------------------------------------------------------
            // check the row indices && print values
            //------------------------------------------------------------------

            if (nzmax > 0 && (Ai == NULL || SPEX_X(A) == NULL))
            {
                // row indices or values not present
                SPEX_PR1 ("i or x invalid\n") ;
                return (SPEX_INCORRECT_INPUT) ;
            }

            // allocate workspace to check for duplicates
            work = (int64_t *) SPEX_calloc (m, sizeof (int64_t)) ;
            if (work == NULL)
            {
                // out of memory
                SPEX_PR1 ("out of memory\n") ;
                SPEX_FREE_ALL;
                return (SPEX_OUT_OF_MEMORY) ;
            }

            for (j = 0 ; j < n ; j++)  // iterate across columns
            {
                SPEX_PR_LIMIT ;
                SPEX_PR2 ("column %"PRId64" :\n", j) ;
                int64_t marked = j+1 ;
                for (p = Ap [j] ; p < Ap [j+1] ; p++)
                {
                    i = Ai [p] ;
                    if (i < 0 || i >= m)
                    {
                        // row indices out of range
                        SPEX_PR1 ("index out of range: (%ld,%ld)\n", i, j) ;
                        SPEX_FREE_ALL ;
                        return (SPEX_INCORRECT_INPUT) ;
                    }
                    else if (work [i] == marked)
                    {
                        // duplicate
                        SPEX_PR1 ("duplicate index: (%ld,%ld)\n", i, j) ;
                        SPEX_FREE_ALL ;
                        return (SPEX_INCORRECT_INPUT) ;
                    }
                    if (pr >= 2)
                    {
                        SPEX_PR_LIMIT ;
                        SPEX_PR2 ("  row %"PRId64" : ", i) ;

                        switch ( A->type)
                        {
                            case SPEX_MPZ:
                            {
                                status = SPEX_mpfr_asprintf(&buff, "%Zd \n",
                                    A->x.mpz[p]);
                                if (status >= 0)
                                {
                                    SPEX_PR2("%s", buff);
                                    SPEX_mpfr_free_str (buff);
                                }
                                break;
                            }
                            case SPEX_MPQ:
                            {
                                status = SPEX_mpfr_asprintf(&buff,"%Qd \n",
                                    A->x.mpq[p]);
                                if (status >= 0)
                                {
                                    SPEX_PR2("%s", buff);
                                    SPEX_mpfr_free_str (buff);
                                }
                                break;
                            }
                            case SPEX_MPFR:
                            {
                                status = SPEX_mpfr_asprintf(&buff, "%.*Rf \n",
                                    prec, A->x.mpfr [p]);
                                if (status >= 0) 
                                {
                                    SPEX_PR2("%s", buff);
                                    SPEX_mpfr_free_str (buff);
                                }
                                break;
                            }
                            case SPEX_FP64:
                            {
                                SPEX_PR2 ("%lf \n", A->x.fp64[p]);
                                break;
                            }
                            case SPEX_INT64:
                            {
                                SPEX_PR2 ("%ld \n", A->x.int64[p]);
                                break;
                            }
                        }
                        if (status < 0)
                        {
                            SPEX_FREE_ALL ;
                            SPEX_PRINTF (" error: %d\n", status) ;
                            return (status) ;
                        }
                    }
                    work [i] = marked ;
                }
            }
        }
        break;

        //----------------------------------------------------------------------
        // check a matrix in triplet format
        //----------------------------------------------------------------------

        case SPEX_TRIPLET:
        {

            int64_t* Aj = A->j;
            int64_t* Ai = A->i;

            //------------------------------------------------------------------
            // basic pointer checking
            //------------------------------------------------------------------

            if (nzmax > 0 && (Ai == NULL || Aj == NULL || SPEX_X(A) == NULL))
            {
                // row indices or values not present
                SPEX_PR1 ("i or j or x invalid\n") ;
                return (SPEX_INCORRECT_INPUT) ;
            }

            //------------------------------------------------------------------
            // print each entry as "Ai Aj Ax"
            //------------------------------------------------------------------

            for (p = 0 ; p < nz ; p++)
            {
                i = Ai[p];
                j = Aj[p];
                if (i < 0 || i >= m || j < 0 || j >= n)
                {
                    // row indices out of range
                    SPEX_PR1 ("invalid index\n") ;
                    SPEX_FREE_ALL ;
                    return (SPEX_INCORRECT_INPUT) ;
                }
                if (pr >= 2)
                {
                    SPEX_PR_LIMIT ;
                    SPEX_PR2 ("  %"PRId64" %"PRId64" : ", i, j) ;

                    switch ( A->type)
                    {
                        case SPEX_MPZ:
                        {
                            status = SPEX_mpfr_asprintf(&buff, "%Zd \n",
                                A->x.mpz [p]);
                            if (status >= 0) 
                            {
                                SPEX_PR2("%s", buff);
                                SPEX_mpfr_free_str (buff);
                            }
                            break;
                        }
                        case SPEX_MPQ:
                        {
                            status = SPEX_mpfr_asprintf (&buff,"%Qd \n",
                                A->x.mpq [p]);
                            if (status >= 0)  
                            {   
                                SPEX_PR2("%s", buff); 
                                SPEX_mpfr_free_str (buff); 
                            }
                            break;
                        }
                        case SPEX_MPFR:
                        {
                            status = SPEX_mpfr_asprintf(&buff, "%.*Rf \n",
                                prec, A->x.mpfr [p]);
                            if (status >= 0)  
                            {   
                                SPEX_PR2("%s", buff); 
                                SPEX_mpfr_free_str (buff); 
                            }
                            break;
                        }
                        case SPEX_FP64:
                        {
                            SPEX_PR2 ("%lf \n", A->x.fp64[p]);
                            break;
                        }
                        case SPEX_INT64:
                        {
                            SPEX_PR2 ("%ld \n", A->x.int64[p]);
                            break;
                        }
                    }
                    if (status < 0)
                    {
                        SPEX_FREE_ALL ;
                        SPEX_PRINTF (" error: %d\n", status) ;
                        return (status) ;
                    }
                }
            }

            //------------------------------------------------------------------
            // check for duplicates
            //------------------------------------------------------------------

            // allocate workspace to check for duplicates
            work = (int64_t *) SPEX_malloc (nz * 2 * sizeof (int64_t)) ;
            if (work == NULL)
            {
                // out of memory
                SPEX_PR1 ("out of memory\n") ;
                SPEX_FREE_ALL;
                return (SPEX_OUT_OF_MEMORY) ;
            }

            // load the (i,j) indices of the triplets into the workspace
            for (p = 0 ; p < nz ; p++)
            {
                work [2*p  ] = Aj [p] ;
                work [2*p+1] = Ai [p] ;
            }

            // sort the (i,j) indices
            qsort (work, nz, 2 * sizeof (int64_t), compar) ;

            // check for duplicates
            for (p = 1 ; p < nz ; p++)
            {
                int64_t this_j = work [2*p  ] ;
                int64_t this_i = work [2*p+1] ;
                int64_t last_j = work [2*(p-1)  ] ;
                int64_t last_i = work [2*(p-1)+1] ;
                if (this_j == last_j && this_i == last_i)
                {
                    SPEX_PR1 ("duplicate index: (%ld, %ld)\n", this_i, this_j) ;
                    SPEX_FREE_ALL ;
                    return (SPEX_INCORRECT_INPUT) ;
                }
            }

        }
        break;

        //----------------------------------------------------------------------
        // check a matrix in dense format
        //----------------------------------------------------------------------

        case SPEX_DENSE:
        {
            // If A is dense, A->i, A->j etc are all NULL. All we must do is
            // to check that its dimensions are correct and print the values if
            // desired.

            if (nzmax > 0 && SPEX_X(A) == NULL)
            {
                // row indices or values not present
                SPEX_PR1 ("x invalid\n") ;
                return (SPEX_INCORRECT_INPUT) ;
            }

            //------------------------------------------------------------------
            // print values
            //------------------------------------------------------------------

            for (j = 0 ; j < n ; j++)
            {
                SPEX_PR_LIMIT ;
                SPEX_PR2 ("column %"PRId64" :\n", j) ;
                for (i = 0; i < m; i++)
                {
                    if (pr >= 2)
                    {
                        SPEX_PR_LIMIT ;
                        SPEX_PR2 ("  row %"PRId64" : ", i) ;

                        switch ( A->type)
                        {
                            case SPEX_MPZ:
                            {
                                status = SPEX_mpfr_asprintf (&buff, "%Zd \n" ,
                                    SPEX_2D(A, i, j, mpz)) ;
                                if (status >= 0)  
                                {   
                                    SPEX_PR2("%s", buff); 
                                    SPEX_mpfr_free_str (buff); 
                                }
                                break;
                            }
                            case SPEX_MPQ:
                            {
                                status = SPEX_mpfr_asprintf (&buff, "%Qd \n",
                                    SPEX_2D(A, i, j, mpq));
                                if (status >= 0)   
                                {    
                                    SPEX_PR2("%s", buff);  
                                    SPEX_mpfr_free_str (buff);  
                                }
                                break;
                            }
                            case SPEX_MPFR:
                            {
                                status = SPEX_mpfr_asprintf (&buff, "%.*Rf \n",
                                    prec, SPEX_2D(A, i, j, mpfr));
                                if (status >= 0)   
                                {    
                                    SPEX_PR2("%s", buff);  
                                    SPEX_mpfr_free_str (buff);  
                                }
                                break;
                            }
                            case SPEX_FP64:
                            {
                                SPEX_PR2 ("%lf \n", SPEX_2D(A, i, j, fp64));
                                break;
                            }
                            case SPEX_INT64:
                            {
                                SPEX_PR2 ("%ld \n", SPEX_2D(A, i, j, int64));
                                break;
                            }
                        }
                        if (status < 0)
                        {
                            SPEX_PR2 (" error: %d\n", status) ;
                            return (status) ;
                        }
                    }
                }
            }
        }
        break;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    SPEX_FREE_ALL ;
    return (SPEX_OK) ;
}

