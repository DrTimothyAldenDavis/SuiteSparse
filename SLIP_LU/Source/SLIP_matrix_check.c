//------------------------------------------------------------------------------
// SLIP_LU/SLIP_matrix_check: check if a matrix is OK
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#define SLIP_FREE_ALL    \
    SLIP_FREE (work) ;

#include "slip_internal.h"

// if pr == 2, turn off printing after 30 lines of output
#define SLIP_PR_LIMIT                       \
    lines++ ;                               \
    if (pr == 2 && lines > 30)              \
    {                                       \
        SLIP_PRINTF ("    ...\n") ;         \
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

/* check the validity of a SLIP_matrix */

// print_level from option struct:
//      0: nothing
//      1: just errors
//      2: errors and terse output
//      3: verbose

SLIP_info SLIP_matrix_check     // returns a SLIP_LU status code
(
    const SLIP_matrix *A,     // matrix to check
    const SLIP_options* option
)
{

    if (!slip_initialized ( )) return (SLIP_PANIC) ;

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    SLIP_info status = 0 ;
    char * buff = NULL ;
    int pr = SLIP_OPTION_PRINT_LEVEL (option) ;
    int64_t nz = SLIP_matrix_nnz(A, option);    // Number of nonzeros in A
    if (nz < 0)
    {
        SLIP_PR1 ("A is NULL or nz invalid\n") ;
        return (SLIP_INCORRECT_INPUT) ;
    }

    int64_t m = A->m ;
    int64_t n = A->n ;
    int64_t nzmax = A->nzmax ;

    if (m < 0)
    {
        SLIP_PR1 ("m invalid\n") ;
        return (SLIP_INCORRECT_INPUT) ;
    }
    if (n < 0)
    {
        SLIP_PR1 ("n invalid\n") ;
        return (SLIP_INCORRECT_INPUT) ;
    }
    if (nzmax < 0)
    {
        SLIP_PR1 ("nzmax invalid\n") ;
        return (SLIP_INCORRECT_INPUT) ;
    }

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    if (A->type < SLIP_MPZ || A->type > SLIP_FP64)
    //  A->kind < SLIP_CSC || A->kind > SLIP_DENSE // checked in SLIP_matrix_nnz
    {
        SLIP_PR1 ("A has invalid type.\n") ;
        return (SLIP_INCORRECT_INPUT) ;
    }

    SLIP_PR2 ("SLIP_matrix: nrows: %"PRId64", ncols: %"PRId64", nz:"
        "%"PRId64", nzmax: %"PRId64", kind: %s, type: %s\n", m, n, nz,
        nzmax, A->kind < 1 ? "CSC" : A->kind < 2 ? "Triplet" : "Dense",
        A->type < 1 ? "MPZ" : A->type < 2 ? "MPQ" : A->type < 3 ?
        "MPFR" : A->type < 4 ? "int64" : "double") ;

    if (pr >= 2)
    {
        SLIP_PR2 ("scale factor: ") ;
        status = SLIP_mpfr_asprintf (&buff,"%Qd\n", A->scale) ;
        if (status >= 0)
        {
            SLIP_PR2 ("%s", buff) ;
            SLIP_mpfr_free_str (buff) ;
        }
    }

    //--------------------------------------------------------------------------
    // initialize workspace
    //--------------------------------------------------------------------------

    int64_t i, j, p, pend ;
    int64_t* work = NULL;   // used for checking duplicates for CSC and triplet
    uint64_t prec = SLIP_OPTION_PREC (option);

    int64_t lines = 0 ;     // # of lines printed so far

    //--------------------------------------------------------------------------
    // check each kind of matrix: CSC, triplet, or dense
    //--------------------------------------------------------------------------

    switch (A->kind)
    {

        //----------------------------------------------------------------------
        // check a matrix in CSC format
        //----------------------------------------------------------------------

        case SLIP_CSC:
        {
            int64_t* Ap = A->p;
            int64_t* Ai = A->i;

            //------------------------------------------------------------------
            // check the column pointers
            //------------------------------------------------------------------

            if (nzmax > 0 && (Ap == NULL || Ap [0] != 0))
            {
                // column pointers invalid
                SLIP_PR1 ("p invalid\n") ;
                return (SLIP_INCORRECT_INPUT) ;
            }
            for (j = 0 ; j < n ; j++)
            {
                p = Ap [j] ;
                pend = Ap [j+1] ;
                if (pend < p || pend > nz)
                {
                    // column pointers not monotonically non-decreasing
                    SLIP_PR1 ("p invalid\n") ;
                    return (SLIP_INCORRECT_INPUT) ;
                }
            }

            //------------------------------------------------------------------
            // check the row indices && print values
            //------------------------------------------------------------------

            if (nzmax > 0 && (Ai == NULL || SLIP_X(A) == NULL))
            {
                // row indices or values not present
                SLIP_PR1 ("i or x invalid\n") ;
                return (SLIP_INCORRECT_INPUT) ;
            }

            // allocate workspace to check for duplicates
            work = (int64_t *) SLIP_calloc (m, sizeof (int64_t)) ;
            if (work == NULL)
            {
                // out of memory
                SLIP_PR1 ("out of memory\n") ;
                SLIP_FREE_ALL;
                return (SLIP_OUT_OF_MEMORY) ;
            }

            for (j = 0 ; j < n ; j++)  // iterate across columns
            {
                SLIP_PR_LIMIT ;
                SLIP_PR2 ("column %"PRId64" :\n", j) ;
                int64_t marked = j+1 ;
                for (p = Ap [j] ; p < Ap [j+1] ; p++)
                {
                    i = Ai [p] ;
                    if (i < 0 || i >= m)
                    {
                        // row indices out of range
                        SLIP_PR1 ("index out of range: (%ld,%ld)\n", i, j) ;
                        SLIP_FREE_ALL ;
                        return (SLIP_INCORRECT_INPUT) ;
                    }
                    else if (work [i] == marked)
                    {
                        // duplicate
                        SLIP_PR1 ("duplicate index: (%ld,%ld)\n", i, j) ;
                        SLIP_FREE_ALL ;
                        return (SLIP_INCORRECT_INPUT) ;
                    }
                    if (pr >= 2)
                    {
                        SLIP_PR_LIMIT ;
                        SLIP_PR2 ("  row %"PRId64" : ", i) ;

                        switch ( A->type)
                        {
                            case SLIP_MPZ:
                            {
                                status = SLIP_mpfr_asprintf(&buff, "%Zd \n",
                                    A->x.mpz[p]);
                                if (status >= 0)
                                {
                                    SLIP_PR2("%s", buff);
                                    SLIP_mpfr_free_str (buff);
                                }
                                break;
                            }
                            case SLIP_MPQ:
                            {
                                status = SLIP_mpfr_asprintf(&buff,"%Qd \n",
                                    A->x.mpq[p]);
                                if (status >= 0)
                                {
                                    SLIP_PR2("%s", buff);
                                    SLIP_mpfr_free_str (buff);
                                }
                                break;
                            }
                            case SLIP_MPFR:
                            {
                                status = SLIP_mpfr_asprintf(&buff, "%.*Rf \n",
                                    prec, A->x.mpfr [p]);
                                if (status >= 0) 
                                {
                                    SLIP_PR2("%s", buff);
                                    SLIP_mpfr_free_str (buff);
                                }
                                break;
                            }
                            case SLIP_FP64:
                            {
                                SLIP_PR2 ("%lf \n", A->x.fp64[p]);
                                break;
                            }
                            case SLIP_INT64:
                            {
                                SLIP_PR2 ("%ld \n", A->x.int64[p]);
                                break;
                            }
                        }
                        if (status < 0)
                        {
                            SLIP_FREE_ALL ;
                            SLIP_PRINTF (" error: %d\n", status) ;
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

        case SLIP_TRIPLET:
        {

            int64_t* Aj = A->j;
            int64_t* Ai = A->i;

            //------------------------------------------------------------------
            // basic pointer checking
            //------------------------------------------------------------------

            if (nzmax > 0 && (Ai == NULL || Aj == NULL || SLIP_X(A) == NULL))
            {
                // row indices or values not present
                SLIP_PR1 ("i or j or x invalid\n") ;
                return (SLIP_INCORRECT_INPUT) ;
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
                    SLIP_PR1 ("invalid index\n") ;
                    SLIP_FREE_ALL ;
                    return (SLIP_INCORRECT_INPUT) ;
                }
                if (pr >= 2)
                {
                    SLIP_PR_LIMIT ;
                    SLIP_PR2 ("  %"PRId64" %"PRId64" : ", i, j) ;

                    switch ( A->type)
                    {
                        case SLIP_MPZ:
                        {
                            status = SLIP_mpfr_asprintf(&buff, "%Zd \n",
                                A->x.mpz [p]);
                            if (status >= 0) 
                            {
                                SLIP_PR2("%s", buff);
                                SLIP_mpfr_free_str (buff);
                            }
                            break;
                        }
                        case SLIP_MPQ:
                        {
                            status = SLIP_mpfr_asprintf (&buff,"%Qd \n",
                                A->x.mpq [p]);
                            if (status >= 0)  
                            {   
                                SLIP_PR2("%s", buff); 
                                SLIP_mpfr_free_str (buff); 
                            }
                            break;
                        }
                        case SLIP_MPFR:
                        {
                            status = SLIP_mpfr_asprintf(&buff, "%.*Rf \n",
                                prec, A->x.mpfr [p]);
                            if (status >= 0)  
                            {   
                                SLIP_PR2("%s", buff); 
                                SLIP_mpfr_free_str (buff); 
                            }
                            break;
                        }
                        case SLIP_FP64:
                        {
                            SLIP_PR2 ("%lf \n", A->x.fp64[p]);
                            break;
                        }
                        case SLIP_INT64:
                        {
                            SLIP_PR2 ("%ld \n", A->x.int64[p]);
                            break;
                        }
                    }
                    if (status < 0)
                    {
                        SLIP_FREE_ALL ;
                        SLIP_PRINTF (" error: %d\n", status) ;
                        return (status) ;
                    }
                }
            }

            //------------------------------------------------------------------
            // check for duplicates
            //------------------------------------------------------------------

            // allocate workspace to check for duplicates
            work = (int64_t *) SLIP_malloc (nz * 2 * sizeof (int64_t)) ;
            if (work == NULL)
            {
                // out of memory
                SLIP_PR1 ("out of memory\n") ;
                SLIP_FREE_ALL;
                return (SLIP_OUT_OF_MEMORY) ;
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
                    SLIP_PR1 ("duplicate index: (%ld, %ld)\n", this_i, this_j) ;
                    SLIP_FREE_ALL ;
                    return (SLIP_INCORRECT_INPUT) ;
                }
            }

        }
        break;

        //----------------------------------------------------------------------
        // check a matrix in dense format
        //----------------------------------------------------------------------

        case SLIP_DENSE:
        {
            // If A is dense, A->i, A->j etc are all NULL. All we must do is
            // to check that its dimensions are correct and print the values if
            // desired.

            if (nzmax > 0 && SLIP_X(A) == NULL)
            {
                // row indices or values not present
                SLIP_PR1 ("x invalid\n") ;
                return (SLIP_INCORRECT_INPUT) ;
            }

            //------------------------------------------------------------------
            // print values
            //------------------------------------------------------------------

            for (j = 0 ; j < n ; j++)
            {
                SLIP_PR_LIMIT ;
                SLIP_PR2 ("column %"PRId64" :\n", j) ;
                for (i = 0; i < m; i++)
                {
                    if (pr >= 2)
                    {
                        SLIP_PR_LIMIT ;
                        SLIP_PR2 ("  row %"PRId64" : ", i) ;

                        switch ( A->type)
                        {
                            case SLIP_MPZ:
                            {
                                status = SLIP_mpfr_asprintf (&buff, "%Zd \n" ,
                                    SLIP_2D(A, i, j, mpz)) ;
                                if (status >= 0)  
                                {   
                                    SLIP_PR2("%s", buff); 
                                    SLIP_mpfr_free_str (buff); 
                                }
                                break;
                            }
                            case SLIP_MPQ:
                            {
                                status = SLIP_mpfr_asprintf (&buff, "%Qd \n",
                                    SLIP_2D(A, i, j, mpq));
                                if (status >= 0)   
                                {    
                                    SLIP_PR2("%s", buff);  
                                    SLIP_mpfr_free_str (buff);  
                                }
                                break;
                            }
                            case SLIP_MPFR:
                            {
                                status = SLIP_mpfr_asprintf (&buff, "%.*Rf \n",
                                    prec, SLIP_2D(A, i, j, mpfr));
                                if (status >= 0)   
                                {    
                                    SLIP_PR2("%s", buff);  
                                    SLIP_mpfr_free_str (buff);  
                                }
                                break;
                            }
                            case SLIP_FP64:
                            {
                                SLIP_PR2 ("%lf \n", SLIP_2D(A, i, j, fp64));
                                break;
                            }
                            case SLIP_INT64:
                            {
                                SLIP_PR2 ("%ld \n", SLIP_2D(A, i, j, int64));
                                break;
                            }
                        }
                        if (status < 0)
                        {
                            SLIP_PR2 (" error: %d\n", status) ;
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

    SLIP_FREE_ALL ;
    return (SLIP_OK) ;
}

