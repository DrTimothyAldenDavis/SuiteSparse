//------------------------------------------------------------------------------
// SLIP_LU/Demo/demos.c: support functions for the demo programs
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// SLIP_print_options: print user options.
// SLIP_process_command_line: process command line for demo programs.
// SLIP_show_usage: show usage of demo program.
// SLIP_tripread: read a matrix from a file in triplet format.
// SLIP_tripread_double: read a double matrix from a file in triplet format.
// SLIP_read_dense: read a dense matrix from a file.

#include "demos.h"

// ignore warnings about unused parameters in this file
#pragma GCC diagnostic ignored "-Wunused-parameter"

//------------------------------------------------------------------------------
// SLIP_print_options
//------------------------------------------------------------------------------

/* Purpose: This function prints out the user specified/default options.
 * this is primarily intended for debugging
 */

void SLIP_print_options // display specified/default options to user
(
    SLIP_options* option    // options (cannot be NULL)
)
{

    char *piv, *order;
    if (option->order == SLIP_COLAMD)
    {
        order = "the COLAMD";
    }
    else if (option->order == SLIP_AMD)
    {
        order = "the AMD";
    }
    else if (option->order == SLIP_NO_ORDERING)
    {
        order = "No";
    }
    else
    {
        order = "(undefined)";
    }

    if (option->pivot == SLIP_SMALLEST)
    {
        piv = "smallest";
    }
    else if (option->pivot == SLIP_DIAGONAL)
    {
        piv = "diagonal";
    }
    else if (option->pivot == SLIP_FIRST_NONZERO)
    {
        piv = "first nonzero";
    }
    else if (option->pivot == SLIP_TOL_SMALLEST)
    {
        piv = "diagonal small tolerance";
    }
    else if (option->pivot == SLIP_TOL_LARGEST)
    {
        piv = "diagonal large tolerance";
    }
    else
    {
        piv = "largest";
    }

    printf("\n\n****COMMAND PARAMETERS****");
    printf("\nUsing %s ordering and selecting the %s pivot", order, piv);
    if (option->pivot == SLIP_TOL_SMALLEST ||
        option->pivot == SLIP_TOL_LARGEST)
    {
        printf("\nTolerance used: %lf\n",option->tol);
    }
}

//------------------------------------------------------------------------------
// SLIP_process_command_line
//------------------------------------------------------------------------------

/* Purpose: This processes the command line for user specified options */

SLIP_info SLIP_process_command_line //processes the command line
(
    int argc,               // number of command line arguments
    char* argv[],           // set of command line arguments
    SLIP_options* option,   // options (cannot be NULL)
    char** mat_name,        // Name of the matrix to be read in
    char** rhs_name,        // Name of the RHS vector to be read in
    SLIP_type *rat,         // data type of output solution:
                            // 1:SLIP_MPZ (default), 2:SLIP_FP64, 3:SLIP_MPFR
    bool *help
)
{
    (*rat) = SLIP_MPZ ;

    (*help) = false ;
    for (int i = 1; i < argc; i++)
    {
        char* arg = (char*) argv[i];
        if ( strcmp(arg,"help") == 0)
        {
            SLIP_show_usage();
            (*help) = true ;
        }
        else if ( strcmp(arg,"p") == 0 || strcmp(arg,"piv") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! There must be a pivot argument between"
                    " 0-5 following p\n");
                return SLIP_INCORRECT_INPUT;
            }
            option->pivot = atoi(argv[i]);
            if (option->pivot < 0 || option->pivot > 5)
            {
                printf("\n****ERROR! Invalid pivot selection!"
                    "\nDefaulting to smallest pivot\n\n");
                option->pivot = SLIP_SMALLEST;
            }
        }
        else if ( strcmp(arg, "q") == 0 || strcmp(arg,"col") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! There must be an argument between 0-2"
                    "following q\n");
                return SLIP_INCORRECT_INPUT;
            }
            option->order = atoi(argv[i]);
            if (option->order < 0 || option->order > 2)
            {
                printf("\n****ERROR! Invalid column ordering"
                    "\nDefaulting to COLAMD\n\n");
                option->order = SLIP_COLAMD;
            }
        }
        else if ( strcmp(arg,"t") == 0 || strcmp(arg, "tol") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! There must be a non negative tolerance"
                    " value following t\n");
                return SLIP_INCORRECT_INPUT;
            }
            else if (!atof(argv[i]))
            {
                printf("\n****ERROR! There must be a non negative tolerance"
                    " value following t\n");
                return SLIP_INCORRECT_INPUT;
            }
            option->tol = atof(argv[i]);
            if (option->tol < 0)
            {
                printf("\n****ERROR! Invalid Tolerance, tolerance must be"
                    " non-negative\n");
                return SLIP_INCORRECT_INPUT;
            }
        }
        else if ( strcmp(arg,"out") == 0 || strcmp(arg, "o") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! o or out must be followed by"
                    " 0 (print nothing) 1 (print err) or 2 (terse) \n");
                return SLIP_INCORRECT_INPUT;
            }
            else if (!atoi(argv[i]))
            {
                printf("\n****ERROR! o or out must be followed by"
                    " 0 (print nothing) 1 (print err) or 2 (terse) \n");
                return SLIP_INCORRECT_INPUT;
            }
            option->print_level = atoi(argv[i]);
        }
        else if ( strcmp(arg, "f") == 0 || strcmp(arg, "file") == 0)
        {
            if (!argv[++i])
            {
                printf("\n****ERROR! Matrix name must be entered\n");
                return SLIP_INCORRECT_INPUT;
            }
            *mat_name = argv[i];
            if (!argv[++i])
            {
                printf("\n****ERROR! Right hand side vector name must"
                    " be entered\n");
                return SLIP_INCORRECT_INPUT;
            }
            *rhs_name = argv[i];
        }
        else
        {
            printf("\n\n**ERROR! Unknown command line parameter: %s"
                    "\nIgnoring this parameter\n",arg);
            return SLIP_INCORRECT_INPUT;
        }
    }

    return SLIP_OK;
}

//------------------------------------------------------------------------------
// SLIP_show_usage
//------------------------------------------------------------------------------

/* Purpose: This function shows the usage of the code.*/
void SLIP_show_usage() //display the usage of the code
{
    printf("\n"
    "\n./SLIP_LU followed by:"
    "\n   c: indicates the solution will be checked"
    "\n   p (or piv) 0 to 5 : indicates the type of pivoting"
    "\n   col or q: column order used: 0: none, 1: COLAMD, 2: AMD"
    "\n   t or tol: tolerance parameter"
    "\n   o or out: level of output printed to screen"
    "\n   f or file: filenames. must be of format MATRIX_NAME RHS_NAME"
    "\nRefer to SLIP_LU/Doc for detailed description of input parameters."
    "\n");
}

//------------------------------------------------------------------------------
// SLIP_tripread
//------------------------------------------------------------------------------

/* Purpose: This function reads in a matrix stored in a triplet format
 * This format used can be seen in any of the example mat files.
 *
 * The first line of the file contains three integers: m, n, nnz,
 * where the matrix is m-by-n with nnz entries.
 *
 * This is followed by nnz lines, each containing a single triplet: i, j, aij,
 * which defines the row index (i), column index (j), and value (aij) of
 * the entry A(i,j).  The value aij is an integer.
 */

SLIP_info SLIP_tripread
(
    SLIP_matrix **A_handle,      // Matrix to be constructed
    FILE* file,                  // file to read from (must already be open)
    SLIP_options* option         // Command options
)
{

    SLIP_info info ;
    if (A_handle == NULL || file == NULL)
    {
        printf ("invalid input\n") ;
        return SLIP_INCORRECT_INPUT;
    }

    (*A_handle) = NULL ;

    int64_t m, n, nz;

    // Read in size of matrix & number of nonzeros
    int s = fscanf(file, "%"PRId64" %"PRId64" %"PRId64"\n", &m, &n, &nz);
    if (feof(file) || s < 3)
    {
        printf ("premature end-of-file\n") ;
        return SLIP_INCORRECT_INPUT;
    }

    // Allocate memory for A
    // A is a triplet mpz_t matrix
    SLIP_matrix* A = NULL;
    info = SLIP_matrix_allocate(&A, SLIP_TRIPLET, SLIP_MPZ, m, n, nz,
        false, true, option);
    if (info != SLIP_OK)
    {
        return (info) ;
    }

    // Read in first values of A
    info = SLIP_gmp_fscanf(file, "%"PRId64" %"PRId64" %Zd\n",
        &A->i[0], &A->j[0], &A->x.mpz[0]);
    if (feof (file) || info != SLIP_OK)
    {
        printf ("premature end-of-file\n") ;
        SLIP_matrix_free(&A, option);
        return SLIP_INCORRECT_INPUT;
    }
    
    // Matrices in this format are 1 based, so we decrement by 1 to get
    // 0 based for internal functions
    A->i[0] -= 1;
    A->j[0] -= 1;
    
    // Read in the values from file
    for (int64_t p = 1; p < nz; p++)
    {
        info = SLIP_gmp_fscanf(file, "%"PRId64" %"PRId64" %Zd\n",
            &A->i[p], &A->j[p], &A->x.mpz[p]);
        if ((feof(file) && p != nz-1) || info != SLIP_OK)
        {
            printf ("premature end-of-file\n") ;
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
        // Conversion from 1 based to 0 based if necessary
        A->i[p] -= 1;
        A->j[p] -= 1;
    }

    // the triplet matrix now has nz entries
    A->nz = nz;

    // A now contains our input matrix in triplet format. We now
    // do a matrix copy to get it into CSC form
    // C is a copy of A which is CSC and mpz_t
    SLIP_matrix* C = NULL;
    SLIP_matrix_copy(&C, SLIP_CSC, SLIP_MPZ, A, option);

    // Free A, set A_handle
    SLIP_matrix_free(&A, option);
    (*A_handle) = C;
    return (info) ;
}

//------------------------------------------------------------------------------
// SLIP_tripread_double
//------------------------------------------------------------------------------

/* Purpose: This function reads in a double matrix stored in a triplet format
 * This format used can be seen in any of the example mat files.
 *
 * The first line of the file contains three integers: m, n, nnz,
 * where the matrix is m-by-n with nnz entries.
 *
 * This is followed by nnz lines, each containing a single triplet: i, j, aij,
 * which defines the row index (i), column index (j), and value (aij) of
 * the entry A(i,j).  The value aij is a floating-point number.
 */

SLIP_info SLIP_tripread_double
(
    SLIP_matrix **A_handle,     // Matrix to be populated
    FILE* file,                 // file to read from (must already be open)
    SLIP_options* option
)
{

    SLIP_info info ;
    if (A_handle == NULL || file == NULL)
    {
        printf ("invalid input\n") ;
        return SLIP_INCORRECT_INPUT;
    }
    (*A_handle) = NULL ;

    // Read in triplet form first
    int64_t m, n, nz;

    // Read in size of matrix & number of nonzeros
    int s = fscanf(file, "%"PRId64" %"PRId64" %"PRId64"\n", &m, &n, &nz);
    if (feof(file) || s < 3)
    {
        printf ("premature end-of-file\n") ;
        return SLIP_INCORRECT_INPUT;
    }

    // First, we create our A matrix which is triplet double
    SLIP_matrix *A = NULL;
    info = SLIP_matrix_allocate(&A, SLIP_TRIPLET, SLIP_FP64, m, n, nz,
        false, true, option);
    if (info != SLIP_OK)
    {
        return (info) ;
    }

    info = fscanf (file, "%"PRId64" %"PRId64" %lf\n",
        &(A->i[0]), &(A->j[0]), &(A->x.fp64[0])) ;
    if (feof(file) || info != SLIP_OK)
    {
        printf ("premature end-of-file\n") ;
        SLIP_matrix_free(&A, option);
        return SLIP_INCORRECT_INPUT;
    }

    // Matrices in this format are 1 based. We decrement
    // the indices by 1 to use internally
    A->i[0] -= 1;
    A->j[0] -= 1;

    // Read in the values from file
    for (int64_t k = 1; k < nz; k++)
    {
        s = fscanf(file, "%"PRId64" %"PRId64" %lf\n",
            &(A->i[k]), &(A->j[k]), &(A->x.fp64[k]));
        if ((feof(file) && k != nz-1) || s < 3)
        {
            printf ("premature end-of-file\n") ;
            SLIP_matrix_free(&A, option);
            return SLIP_INCORRECT_INPUT;
        }
        // Conversion from 1 based to 0 based
        A->i[k] -= 1;
        A->j[k] -= 1;
    }

    // the triplet matrix now has nz entries
    A->nz = nz;

    // At this point, A is a double triplet matrix. We make a copy of it with C

    SLIP_matrix* C = NULL;
    SLIP_matrix_copy(&C, SLIP_CSC, SLIP_MPZ, A, option);

    // Success. Set A_handle = C and free A

    SLIP_matrix_free(&A, option);
    (*A_handle) = C;
    return (info) ;
}

//------------------------------------------------------------------------------
// SLIP_read_dense
//------------------------------------------------------------------------------

/* Purpose: Read a dense matrix for RHS vectors. 
 * the values in the file must be integers
 */

SLIP_info SLIP_read_dense
(
    SLIP_matrix **b_handle, // Matrix to be constructed
    FILE* file,             // file to read from (must already be open)
    SLIP_options* option
)
{

    if (file == NULL)
    {
        printf ("invalid inputs\n") ;
        return SLIP_INCORRECT_INPUT;
    }
    int64_t nrows, ncols;
    SLIP_info info ;

    // First, we obtain the dimension of the matrix
    int s = fscanf(file, "%"PRId64" %"PRId64, &nrows, &ncols) ;
    if (feof(file) || s < 2)
    {
        printf ("premature end-of-file\n") ;
        return SLIP_INCORRECT_INPUT;
    }

    // Now, we create our dense mpz_t matrix
    SLIP_matrix* A = NULL;
    info = SLIP_matrix_allocate(&A, SLIP_DENSE, SLIP_MPZ, nrows, ncols,
        nrows*ncols, false, true, option);
    if (info != SLIP_OK)
    {
        return (info) ;
    }

    // We now populate the matrix b.
    for (int64_t i = 0; i < nrows; i++)
    {
        for (int64_t j = 0; j < ncols; j++)
        {
            info = SLIP_gmp_fscanf(file, "%Zd", &(SLIP_2D(A, i, j, mpz)));
            if (info != SLIP_OK)
            {
                printf("\n\nhere at i = %"PRId64" and j = %"PRId64"", i, j);
                return SLIP_INCORRECT_INPUT;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Success, set b_handle = A
    //--------------------------------------------------------------------------

    (*b_handle) = A;
    return (info) ;
}
