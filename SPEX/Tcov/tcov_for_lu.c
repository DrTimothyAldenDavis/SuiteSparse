// ----------------------------------------------------------------------------
// SPEX/Tcov/tcov_for_lu.c: test coverage for SPEX_LU
// ----------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//-----------------------------------------------------------------------------

/*
 * When the test is run without input argument, brutal test is used and simple
 * test otherwise. Read the following for detailed instruction and information
 *
 * For simple test, the test needs to be run with command
 * ./tcov_test Ab_type N list1[0] ... list1[N-1] M list2[0] ... list2[M-1]
 * Ab_type: type of original matrix A and vector b: 0 mpz, 1 mpq, 2 mpfr, 3
 *     int64, 4 double. For Ab_type >= 5, it corresponds to 15 type of original
 *     matrix A (i.e., (csc, triplet, dense) x (mpz, mpq, mpfr, int64, double)),
 *     specifically, A->type=(Ab_type-5)%5, and A->kind=(Ab_type-5)/5.
 * N and list1 specify the test list for spex_gmp_ntrials (in SPEX_gmp.c)
 * M and list2 specify the test list for malloc_count (in tcov_utilities.h)
 * N, list1, M, list2 are optional, but N and list1 are required when M and
 * list2 is wanted
 *
 * For brutal test, the test is run with command
 * ./tcov_test
 * the test will run through all cases
 * (specifically, Ab_type={0, 1, 2, 3, 4, 5,...,20})
 * each case run from malloc_count = 0 to a number that can guarantee
 * malloc_count > 0 when the case finishes
 */

/* For simple test ONLY!!
 * uncomment to show the input lists for spex_gmp_ntrials and malloc_count
 */
// #define SPEX_TCOV_SHOW_LIST

/* This program will exactly solve the sparse linear system A*x = b by
 * performing the SPEX Left LU factorization. Refer to README.txt for
 * information on how to properly use this code.
 */

#define SPEX_FREE_ALL                              \
{                                                  \
    OK (SPEX_matrix_free (&A, option));            \
    OK (SPEX_matrix_free (&b, option));            \
    OK (SPEX_matrix_free (&B, option));            \
    OK (SPEX_matrix_free (&Ax, option));           \
    OK (SPEX_matrix_free (&sol, option));          \
    OK (SPEX_matrix_free (&sol_doub, option));     \
    OK (SPEX_symbolic_analysis_free (&S, option)); \
    SPEX_FREE (option);                            \
    OK (SPEX_finalize ());                         \
}

#include "tcov_utilities.h"

#define MAX_MALLOC_COUNT 1000

int64_t Ap[5] = {0, 3, 5, 8, 11};
int64_t Ai[11]   = {0, 1, 2, 2, 3, 1, 2, 3, 0, 1,  2};
double Axnum[11] = {1, 2, 7, 1, 2, 4, 1, 3, 1, 12, 1};  // Numerator of x
double Axden[11] = {3, 3, 6, 1, 7, 1, 1, 1, 5, 1,  1};  // Denominator of x
double bxnum[4] = {170, 1820, 61, 670};                // Numerator of b
double bxden[4] = {15,  3,   6,  7};                    // Denominator of b
int64_t Axnum3[11] = {1, 2, 7, 1, 2, 4, 1, 3, 1, 12, 1};    // Numerator of x
int64_t Axden3[11] = {3, 3, 6, 1, 7, 1, 1, 1, 5, 1,  1};    // Denominator of x
int64_t bxnum3[4] = {17, 182, 61, 67};                      // Numerator of b
int64_t bxden3[4] = {15,  3,   6,  7};                      // Denominator of b

int main ( int argc, char *argv[])
{
    bool is_simple_test = true;
    int Ab_type = 0;
    int64_t malloc_count_list[20]= { -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1};
    int64_t num_of_trials = 0 ;
    int64_t num_of_malloc_t = 0;
    int64_t *gmp_ntrial_list=NULL;         // only used in simple test
    int64_t *malloc_trials_list=NULL;          // only used in simple test
    bool pretend_to_fail = false ;

    //------------------------------------------------------------------
    // Define variables
    //------------------------------------------------------------------

    // used in as source in different Ab_type for A and b
    SPEX_matrix B  = NULL ;
    SPEX_matrix Ax = NULL ;

    // matrix A, b and solution
    SPEX_matrix A = NULL ;
    SPEX_matrix b = NULL ;
    SPEX_matrix sol = NULL ;
    SPEX_matrix sol_doub = NULL ;

    // Column permutation
    SPEX_symbolic_analysis S = NULL ;

    SPEX_options option = NULL ;

    //--------------------------------------------------------------------------
    // parse input arguments
    //--------------------------------------------------------------------------

    if (argc == 1)                         // brutal test
    {
        is_simple_test = false;
        num_of_trials = 20;
    }
    else                                   // simple test
    {
        is_simple_test = true;

        int64_t arg_count = 0;
        // type of Matrix A and vector b:
        // 0 mpz, 1 double, 2 int64_t, 3 mpq, 4 mpfr
        Ab_type = atoi(argv[++arg_count]);
        if (!argv[++arg_count])
        {
            num_of_trials=1;
            gmp_ntrial_list= malloc (num_of_trials* sizeof(int64_t));
            gmp_ntrial_list[0]=-1;
            arg_count--;
        }
        else
        {
            num_of_trials=atoi(argv[arg_count]);
            gmp_ntrial_list= malloc (num_of_trials* sizeof(int64_t));
            for (int64_t k=0; k<num_of_trials; k++)
            {
                if (argv[++arg_count])
                {
                    gmp_ntrial_list[k]=atoi(argv[arg_count]);
                }
                else
                {
                    num_of_trials=1;
                    gmp_ntrial_list[0]=-1;
                    arg_count--;
                }
            }
        }
        if (!argv[++arg_count])
        {
            num_of_malloc_t=1;
            malloc_trials_list= malloc (num_of_malloc_t* sizeof(int64_t));
            malloc_trials_list[0]=MAX_MALLOC_COUNT;//INT_MAX;
        }
        else
        {
            num_of_malloc_t=atoi(argv[arg_count]);
            malloc_trials_list= malloc (num_of_malloc_t* sizeof(int64_t));
            for (int64_t k=0; k<num_of_malloc_t; k++)
            {
                if (argv[++arg_count])
                {
                    malloc_trials_list[k]=atoi(argv[arg_count]);
                }
                else
                {
                    num_of_malloc_t=1;
                    malloc_trials_list[0]=MAX_MALLOC_COUNT;//INT_MAX;
                }
            }
        }

        #ifdef SPEX_TCOV_SHOW_LIST
        printf ("gmp ntrials list is: ");
        for (int64_t k=0; k<num_of_trials; k++)
        {
            printf("%ld   ",gmp_ntrial_list[k]);
        }
        printf("\nmalloc trial list is: ");
        for (int64_t k=0; k<num_of_malloc_t; k++)
        {
            printf("%d   ",malloc_trials_list[k]);
        }
        printf("\n");
        #endif /* SPEX_TCOV_SHOW_LIST */
    }

    //--------------------------------------------------------------------------
    // test calloc, realloc, free
    //--------------------------------------------------------------------------

    SPEX_info info ;
    info = SPEX_initialize ( ) ;
    OK (SPEX_finalize ( ));
    OK (SPEX_initialize ( ));
    int *p4 = SPEX_calloc (5, sizeof (int));
    TEST_ASSERT (p4 != NULL);
    bool ok ;
    p4 = SPEX_realloc (6, 5, sizeof (int), p4, &ok);
    TEST_ASSERT (ok);
    OK (SPEX_finalize ( ));
    p4 = SPEX_realloc (7, 6, sizeof (int), p4, &ok);
    // TEST_ASSERT (!ok);
    OK (SPEX_initialize ( ));
    SPEX_FREE (p4);
    OK (SPEX_finalize ( ));
    spex_gmp_free (NULL, 0);

    //--------------------------------------------------------------------------
    // SPEX_version
    //--------------------------------------------------------------------------

    int version [3] ;
    char date [128] ;
    OK (SPEX_version (version, date));
    printf ("SPEX version %d.%d.%d (%s)\n",
        version [0], version [1], version [2], date);

    //--------------------------------------------------------------------------
    // run all trials
    //--------------------------------------------------------------------------

    // For SIMPLE_TEST, outer loop iterates for spex_gmp_ntrials initialized
    // from list1 (input for tcov_test) and inner loop interates for
    // malloc_count initialized from list2 (input for tcov_test).
    //
    // For non SIMPLE_TEST, outer loop iterates for Ab_type from 0 to 20, and
    // inner loop iterates for malloc_count initialized from 0 to
    // MAX_MALLOC_COUNT, break when malloc_count>0 at the end of inner loop.

    for (int64_t k=0; k<num_of_trials; k++)
    {
        if (is_simple_test)
        {
            // only the first outter loop will iterate across all list2
            if (k == 1)
            {
                num_of_malloc_t=1;
                malloc_trials_list[0]=INT_MAX;
            }
        }
        else
        {
            Ab_type = k;
            num_of_malloc_t = MAX_MALLOC_COUNT;
        }

        for (int64_t kk=0; kk<num_of_malloc_t; kk++)
        {
            pretend_to_fail = false ;
            if (is_simple_test)
            {
                spex_set_gmp_ntrials (gmp_ntrial_list[k]) ;
                printf("initial spex_gmp_ntrials=%ld\n",spex_get_gmp_ntrials());
                malloc_count=malloc_trials_list[kk];
                printf("%"PRId64" out of %"PRId64", "
                    "initial malloc_count=%"PRId64"\n",
                    kk, num_of_malloc_t, malloc_count);
            }
            else
            {
                malloc_count = kk;
                printf("[Ab_type malloc_count] = [%d %"PRId64"]\n",
                    Ab_type, malloc_count);
            }

            int64_t n=4, numRHS=1, j, nz=11;

            //------------------------------------------------------------------
            // Initialize SPEX Left LU process
            //------------------------------------------------------------------

            info = SPEX_initialize_expert (tcov_malloc, tcov_calloc,
                tcov_realloc, tcov_free);

            if (pretend_to_fail || info != SPEX_OK)
            {
                continue ;
            }

            TEST_CHECK_FAILURE (SPEX_initialize ( ), SPEX_PANIC);
            if (pretend_to_fail) continue ;

            //------------------------------------------------------------------
            // Allocate memory
            //------------------------------------------------------------------

            TEST_CHECK (SPEX_create_default_options (&option));
            if (pretend_to_fail) continue ;
            option->print_level = 3;

            if (Ab_type >= 0 && Ab_type <= 4)
            {

                //--------------------------------------------------------------
                // Solve A*x=b where A and b are created from mpz entries
                //--------------------------------------------------------------

                TEST_ASSERT (B == NULL);
                TEST_ASSERT (Ax == NULL);
                TEST_CHECK(SPEX_matrix_allocate(&B, SPEX_DENSE,
                    (SPEX_type) Ab_type, n, numRHS, n*numRHS, false, true,
                    option));
                if (pretend_to_fail) continue ;

                TEST_CHECK(SPEX_matrix_allocate(&Ax, SPEX_CSC,
                    (SPEX_type) Ab_type, n, n, nz, false, true, option));

                if (pretend_to_fail) continue ;

                // fill Ax->i and Ax->p
                for (j = 0; j < n+1; j++)
                {
                    Ax->p[j] = Ap[j];
                }
                for (j = 0; j < nz; j++)
                {
                    Ax->i[j] = Ai[j];
                }

                // special failure cases
                if (Ab_type == 2)// MPFR
                {
                    // create empty A and b using uninitialized double mat/array
                    // to trigger all-zero array condition
                    TEST_ASSERT (A == NULL);
                    TEST_ASSERT (b == NULL);
                    TEST_ASSERT (sol == NULL);
                    TEST_CHECK(SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, Ax,
                        option));
                    if (pretend_to_fail) continue ;
                    TEST_CHECK(SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ, B,
                        option));
                    if (pretend_to_fail) continue ;
                    // to trigger SPEX_SINGULAR
                    TEST_CHECK_FAILURE(SPEX_lu_backslash(&sol, SPEX_MPQ, A,
                        b, option), SPEX_SINGULAR);
                    if (pretend_to_fail) continue ;
                    option->pivot = SPEX_LARGEST;
                    TEST_CHECK_FAILURE(SPEX_lu_backslash(&sol, SPEX_MPQ, A,
                        b, option), SPEX_SINGULAR);
                    if (pretend_to_fail) continue ;
                    option->pivot = SPEX_FIRST_NONZERO;
                    TEST_CHECK_FAILURE(SPEX_lu_backslash(&sol, SPEX_MPQ, A,
                       b, option), SPEX_SINGULAR);
                    if (pretend_to_fail) continue ;

                    //free the memory alloc'd
                    OK (SPEX_matrix_free (&A, option));
                    OK (SPEX_matrix_free (&b, option));
                    TEST_ASSERT (A == NULL);
                    TEST_ASSERT (b == NULL);
                    TEST_ASSERT (sol == NULL);

                    // trigger gcd == 1
                    int32_t prec = option->prec;
                    option->prec = 17;
                    double pow2_17 = pow(2,17);
                    TEST_ASSERT (B != NULL);
                    for (j = 0; j < n && !pretend_to_fail; j++)        // Get B
                    {
                        TEST_CHECK(SPEX_mpfr_set_d(SPEX_2D(B,j,0,mpfr),
                            bxnum[j]/pow2_17, MPFR_RNDN));
                    }
                    if (pretend_to_fail) continue ;
                    TEST_CHECK(SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ,B,
                        option));
                    if (pretend_to_fail) continue ;
                    OK (SPEX_matrix_free (&b, option));
                    TEST_ASSERT (b == NULL);

                    // restore default precision
                    option->prec = prec;

                    // use diagonal entries as pivot
                    option->pivot = SPEX_DIAGONAL;
                }
                else if (Ab_type == 4)// double
                {
                    // create empty A using uninitialized double mat/array
                    // to trigger all-zero array condition
                    TEST_ASSERT (Ax != NULL);
                    TEST_ASSERT (A == NULL);
                    TEST_CHECK(SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, Ax,
                        option));
                    if (pretend_to_fail) continue ;
                    OK (SPEX_matrix_free (&A, option));
                    TEST_ASSERT (A == NULL);

                    // trigger gcd == 1
                    TEST_ASSERT (B != NULL);
                    for (j = 0; j < n; j++)                           // Get b
                    {
                        SPEX_2D(B,j,0,fp64) = bxnum[j]/1e17;
                    }
                    TEST_ASSERT (b == NULL);
                    TEST_CHECK(SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ, B,
                        option));
                    if (pretend_to_fail) continue ;
                    OK (SPEX_matrix_free (&b, option));
                    TEST_ASSERT (b == NULL);
                    if (pretend_to_fail) continue ;

                    // use smallest entry as pivot
                    option->pivot = SPEX_SMALLEST;
                }

                // fill Ax->x and b->x
                for (j = 0; j < n && !pretend_to_fail; j++)           // Get b
                {
                    TEST_ASSERT (B != NULL);
                    if (Ab_type == 0) //MPZ
                    {
                        TEST_CHECK(SPEX_mpz_set_ui(SPEX_2D(B, j, 0, mpz),
                            bxnum3[j]));
                        if (pretend_to_fail) break ;
                    }
                    else if (Ab_type == 1)// MPQ
                    {
                        TEST_CHECK(SPEX_mpq_set_ui(SPEX_2D(B,j,0,mpq),
                            bxnum3[j], bxden3[j]));
                        if (pretend_to_fail) break ;
                    }
                    else if (Ab_type == 2)// MPFR
                    {
                        TEST_CHECK(SPEX_mpfr_set_d(SPEX_2D(B,j,0,mpfr),bxnum[j],
                            MPFR_RNDN));
                        if (pretend_to_fail) break ;
                        TEST_CHECK(SPEX_mpfr_div_d(SPEX_2D(B,j,0,mpfr),
                            SPEX_2D(B,j,0,mpfr), bxden[j], MPFR_RNDN));
                        if (pretend_to_fail) break ;
                    }
                    else if (Ab_type == 3)// INT64
                    {
                        SPEX_2D(B,j,0,int64)=bxnum3[j];
                    }
                    else // double
                    {
                        SPEX_2D(B,j,0,fp64) = bxnum[j];
                    }
                }
                if (pretend_to_fail) continue ;

                for (j = 0; j < nz; j++)                          // Get Ax
                {
                    TEST_ASSERT (Ax != NULL);
                    if (Ab_type == 0)
                    {
                        TEST_CHECK(SPEX_mpz_set_ui(Ax->x.mpz[j],Axnum3[j]));
                        if (pretend_to_fail) break ;
                    }
                    else if (Ab_type == 1)
                    {
                        TEST_CHECK(SPEX_mpq_set_ui(Ax->x.mpq[j],Axnum3[j],
                            Axden3[j]));
                        if (pretend_to_fail) break ;
                    }
                    else if (Ab_type == 2)
                    {
                        TEST_CHECK(SPEX_mpfr_set_d(Ax->x.mpfr[j], Axnum[j],
                            MPFR_RNDN));
                        if (pretend_to_fail) break ;
                        TEST_CHECK(SPEX_mpfr_div_d(Ax->x.mpfr[j], Ax->x.mpfr[j],
                            Axden[j], MPFR_RNDN))
                        if (pretend_to_fail) break ;
                    }
                    else if (Ab_type == 3)
                    {
                        Ax->x.int64[j]=Axnum3[j];
                    }
                    else
                    {
                        Ax->x.fp64[j] = Axnum[j]/Axden[j];
                    }
                }
                if (pretend_to_fail) continue ;

                // successful case
                TEST_ASSERT (A == NULL);
                TEST_ASSERT (b == NULL);
                TEST_CHECK(SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, Ax,option));
                if (pretend_to_fail) continue ;
                TEST_CHECK(SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ,B,option));
                if (pretend_to_fail) continue ;
                OK (SPEX_matrix_free(&B, option));
                OK (SPEX_matrix_free(&Ax, option));
                TEST_ASSERT (Ax == NULL);
                TEST_ASSERT (B == NULL);
                if (pretend_to_fail) continue ;
            }
            else // 5 =< Ab_type < 20
            {
                //--------------------------------------------------------------
                // test gmp functions that are not used in SPEX Left LU
                //--------------------------------------------------------------
                /*TEST_CHECK(SPEX_mpz_init(mpz1));
                TEST_CHECK(SPEX_mpz_init(mpz2));
                TEST_CHECK(SPEX_mpz_init(mpz3));
                TEST_CHECK(SPEX_mpz_set_ui(mpz1, 2));
                TEST_CHECK(SPEX_mpz_set_ui(mpz2, 3));

                TEST_CHECK(SPEX_mpz_add(mpz3, mpz2, mpz1));
                TEST_CHECK(SPEX_mpz_addmul(mpz3, mpz2, mpz1));*/

                //--------------------------------------------------------------
                // Test SPEX_matrix_copy and SPEX_matrix_check brutally
                // and some special failure cases
                //--------------------------------------------------------------

                n = 4, nz = 11;
                int64_t m1, n1, nz1;
                int64_t I[11]={0, 1, 2, 2, 3, 1, 2, 3, 0, 1, 2};
                int64_t J[11]={0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3};
                int64_t P[11]={0, 3, 5, 8, 11};

                double x_doub2[11] = {1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4};
                int64_t x_int64[11] = {1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4};

                // find the type and kind of the source matrix to copy from
                Ab_type = Ab_type > 19 ? 19:Ab_type;
                int tk = Ab_type-5;
                int type = tk%5;
                int kind = tk/5;
                if (kind != 2)
                {
                    m1 = n;
                    n1 = n;
                    nz1 = nz;
                }
                else
                {
                    m1 = nz1;
                    n1 = 1;
                    nz1 = nz1;
                }
                TEST_ASSERT (Ax == NULL);
                TEST_CHECK(SPEX_matrix_allocate(&Ax, (SPEX_kind) kind,
                    (SPEX_type)type, m1, n1, nz1, false, true, option));
                if (pretend_to_fail) continue ;
                if (kind == 1){Ax->nz = nz1;}

                // fill Ax->p
                if(kind == 0)
                {
                    for (j = 0; j < n+1; j++)
                    {
                        Ax->p[j] = P[j];
                    }
                }

                // fill Ax->i and Ax->j
                for (j = 0; j < nz && !pretend_to_fail ; j++)
                {
                    TEST_ASSERT(Ax != NULL);
                    if (kind != 2) {Ax->i[j] = I[j];}
                    // triplet
                    if (kind == 1){  Ax->j[j] = J[j];}
                    switch (type)
                    {
                        case 0: // MPZ
                        {
                            TEST_CHECK(SPEX_mpz_set_si(Ax->x.mpz[j],
                                x_int64[j]));
                        }
                        break;

                        case 1: // MPQ
                        {
                            TEST_CHECK(SPEX_mpq_set_ui(Ax->x.mpq[j],
                                2*x_int64[j],2));
                        }
                        break;

                        case 2: // MPFR
                        {
                            TEST_CHECK(SPEX_mpfr_set_d(Ax->x.mpfr[j],
                                x_doub2[j], MPFR_RNDN));
                            if (!pretend_to_fail)
                            {
                                TEST_CHECK(SPEX_mpfr_div_d(Ax->x.mpfr[j],
                                    Ax->x.mpfr[j], 1, MPFR_RNDN));
                            }
                        }
                        break;

                        case 3: // INT64
                        {
                            Ax->x.int64[j] = x_int64[j];
                        }
                        break;

                        case 4: // double
                        {
                            Ax->x.fp64[j] = x_doub2[j];
                        }
                        break;

                        default: break;
                    }
                    if (pretend_to_fail) break ;
                }
                if (pretend_to_fail) continue;
                TEST_CHECK (SPEX_matrix_check (Ax, option));
                if (pretend_to_fail) continue ;

                // convert to all different type of matrix
                for (int tk1 = 0; tk1 < 15 && !pretend_to_fail; tk1++)
                {
                    // successful cases
                    int type1 = tk1%5;
                    int kind1 = tk1/5;
                    printf("converting from %s(%d) %s(%d) to %s(%d) "
                        "%s(%d)\n",kind < 1 ? "CSC" : kind < 2 ? "Triplet" :
                        "Dense",kind, type < 1 ? "MPZ" : type < 2 ? "MPQ" :
                        type <3 ?  "MPFR" : type < 4 ? "int64" :
                        "double",type, kind1 < 1 ?  "CSC" : kind1 < 2 ?
                        "Triplet" : "Dense", kind1,type1 < 1 ? "MPZ" :
                        type1 < 2 ? "MPQ" : type1 < 3 ?  "MPFR" : type1 < 4
                        ? "int64" : "double",type1);

                    TEST_ASSERT (A == NULL);
                    TEST_CHECK(SPEX_matrix_copy(&A, (SPEX_kind)kind1,
                        (SPEX_type) type1, Ax, option));
                    if (pretend_to_fail) break ;

                    TEST_CHECK (SPEX_matrix_check (A, NULL));
                    if (pretend_to_fail) break ;

                    // just perform once to try some failure cases
                    if (tk == 0 && tk1 == 0)
                    {
                        // fail SPEX_lu_solve
                        TEST_ASSERT (b == NULL);
                        TEST_CHECK(SPEX_matrix_allocate(&b, SPEX_DENSE,
                            SPEX_MPZ, 1, 1, 1, true, true, option));
                        if (pretend_to_fail) break ;
                        TEST_CHECK_FAILURE(SPEX_lu_solve(NULL, NULL,
                            b, option), SPEX_INCORRECT_INPUT);
                        OK (SPEX_matrix_free (&b, option));
                        TEST_ASSERT (b == NULL);
                    }
                    else if (tk == 0 && (tk1 == 1 || tk1 == 2 || tk1 == 4))
                    {
                        // test SPEX_matrix_copy with scale
                        OK (SPEX_matrix_free (&A, option));
                        TEST_ASSERT (A == NULL);
                        if (pretend_to_fail) break ;
                        TEST_CHECK (SPEX_mpq_set_ui (Ax->scale, 2, 5));
                        if (pretend_to_fail) break ;
                        TEST_CHECK(SPEX_matrix_copy(&A, (SPEX_kind)kind1,
                            (SPEX_type) type1, Ax, option));
                        if (pretend_to_fail) break ;
                        TEST_CHECK (SPEX_mpq_set_ui (Ax->scale, 1, 1));
                        if (pretend_to_fail) break ;
                    }
                    else if (tk == 0 && tk1 == 3) //A is CSC int64
                    {
                        // test SPEX_matrix_check
                        A->i[0] = -1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        SPEX_FREE(A->x.int64);
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        A->p[1] = 2;
                        A->p[2] = 1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        A->p[0] = 1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        A->type = -1;// invalid type
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        A->nzmax = -1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        A->m = -1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(NULL, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;

                        // Incorrect calling with NULL pointer(s)
                        TEST_CHECK_FAILURE(SPEX_lu_analyze(NULL,A,NULL),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;

                        // test SPEX_matrix_copy with scale
                        OK (SPEX_matrix_free (&A, option));
                        if (pretend_to_fail) break ;
                        TEST_CHECK (SPEX_mpq_set_ui (Ax->scale, 5, 2));
                        if (pretend_to_fail) break ;
                        TEST_CHECK(SPEX_matrix_copy(&A, (SPEX_kind)kind1,
                            (SPEX_type) type1, Ax, option));
                        if (pretend_to_fail) break ;
                        TEST_CHECK (SPEX_mpq_set_ui (Ax->scale, 1, 1));
                        if (pretend_to_fail) break ;
                    }
                    else if (tk == 0 && tk1 == 8) // A= Triplet int64
                    {
                        // test SPEX_matrix_check
                        A->i[0] = -1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        SPEX_FREE(A->x.int64);
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                        A->n = -1;
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                    }
                    else if (tk == 0 && tk1 == 13)//A= dense int64
                    {
                        SPEX_FREE(A->x.int64);
                        TEST_CHECK_FAILURE(SPEX_matrix_check(A, option),
                            SPEX_INCORRECT_INPUT);
                        if (pretend_to_fail) break ;
                    }
                    else if (tk == 4 && tk1 == 3)// converting double to int64
                    {
                        // test spex_cast_array
                        OK (SPEX_matrix_free (&A, option));
                        if (pretend_to_fail) break ;
                        Ax->x.fp64[0] = 0.0/0.0;//NAN
                        Ax->x.fp64[1] = DBL_MAX; //a value > INT64_MAX;
                        Ax->x.fp64[2] = -DBL_MAX;//a value < INT64_MIN;
                        TEST_CHECK(SPEX_matrix_copy(&A, (SPEX_kind)kind1,
                            (SPEX_type) type1, Ax, option));
                        if (pretend_to_fail) break ;

                        Ax->x.fp64[0] = x_doub2[0];
                        Ax->x.fp64[1] = x_doub2[1];
                        Ax->x.fp64[2] = x_doub2[2];
                    }
                    OK (SPEX_matrix_free (&A, option));
                    if (pretend_to_fail) break ;
                }
                if (pretend_to_fail) continue;

                if (tk == 3)
                {
                    // fail SPEX_matrix_copy
                    TEST_ASSERT (A == NULL);
                    TEST_CHECK_FAILURE(SPEX_matrix_copy(&A, 7,
                        (SPEX_type) type, Ax, option), SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;
                    // failure case: Ax->x = NULL
                    SPEX_FREE(Ax->x.int64);
                    TEST_ASSERT (A == NULL);
                    TEST_CHECK_FAILURE(SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ,
                        Ax,option), SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;

                    // fail SPEX_matrix_allocate
                    TEST_CHECK_FAILURE(SPEX_matrix_allocate(NULL,
                        SPEX_DENSE, SPEX_MPZ, 1, 1, 1,
                        true, true, option), SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;
                    TEST_ASSERT (b == NULL);
                    TEST_CHECK_FAILURE(SPEX_matrix_allocate(&b,
                        SPEX_DENSE, SPEX_MPZ, -1, 1, 1,
                        true, true, option), SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;

                    // test SPEX_matrix_allocate
                    TEST_ASSERT (b == NULL);
                    TEST_CHECK(SPEX_matrix_allocate(&b, SPEX_DENSE,
                        SPEX_MPQ, 1, 1, 1, false, false, option));
                    if (pretend_to_fail) continue ;
                    OK (SPEX_matrix_free (&b, option));
                    TEST_ASSERT (b == NULL);
                    TEST_CHECK(SPEX_matrix_allocate(&b, SPEX_DENSE,
                        SPEX_MPFR, 1, 1, 1, false, false, option));
                    if (pretend_to_fail) continue ;
                    OK (SPEX_matrix_free (&b, option));
                    TEST_ASSERT (b == NULL);
                    if (pretend_to_fail) continue ;

                    //test coverage for spex_gmp_reallocate()
                    void *p_new = NULL;
                    TEST_CHECK(spex_gmp_realloc_test(&p_new, NULL,0,1));
                    if (pretend_to_fail) continue ;
                    TEST_CHECK(spex_gmp_realloc_test(&p_new,p_new,1,0));
                    if (pretend_to_fail) continue ;
                }

                //--------------------------------------------------------------
                // test SPEX_matrix_check on a triplet matrix with bad triplets
                //--------------------------------------------------------------

//              printf ("\n[ SPEX_matrix_check -------------------------\n");
                OK (SPEX_matrix_free (&A, option));
                int64_t I2 [4] = { 1, 2, 1, 1 } ;
                int64_t J2 [4] = { 1, 0, 0, 1 } ;
                TEST_ASSERT (A == NULL);
                TEST_CHECK (SPEX_matrix_allocate (&A, SPEX_TRIPLET,
                    SPEX_INT64, 3, 3, 4, true, false, option));
                if (pretend_to_fail) continue ;
                A->i = I2 ;
                A->j = J2 ;
                A->x.int64 = I2 ;
                A->nz = 4 ;
//              printf ("invalid triplet matrix expected:\n");
                TEST_CHECK_FAILURE (SPEX_matrix_check (A, option),
                    SPEX_INCORRECT_INPUT);
                if (pretend_to_fail) continue ;
                OK (SPEX_matrix_free (&A, option));
                TEST_ASSERT (A == NULL);

                TEST_CHECK (SPEX_matrix_allocate (&A, SPEX_CSC,
                    SPEX_INT64, 3, 3, 4, true, false, option));
                if (pretend_to_fail) continue ;
                int64_t P3 [4] = { 0, 2, 4, 4 } ;
                int64_t I3 [4] = { 0, 0, 0, 0 } ;
                A->p = P3 ;
                A->i = I3 ;
                A->x.int64 = I3 ;
//              printf ("invalid CSC matrix expected:\n");
                TEST_CHECK_FAILURE (SPEX_matrix_check (A, option),
                    SPEX_INCORRECT_INPUT);
                if (pretend_to_fail) continue ;
                OK (SPEX_matrix_free (&A, option));
                TEST_ASSERT (A == NULL);
//              printf ("-----------------------------------------------]\n");

                //--------------------------------------------------------------

                SPEX_FREE_ALL;

                // for miscellaneous test, continue to next loop directly
                if (!is_simple_test)
                {
                    if (malloc_count > 0)
                    {
                        malloc_count_list[Ab_type] = kk;
                        break;
                    }
                    else {continue;}
                }
                else
                {
                    continue;
                }
            }

            if (Ab_type%2 == 0)
            {

                //--------------------------------------------------------------
                // SPEX Left LU backslash
                // solve Ax=b in full precision rational arithmetic
                //--------------------------------------------------------------

                TEST_ASSERT (sol == NULL);
                TEST_CHECK(SPEX_lu_backslash(&sol, SPEX_MPQ, A, b,option));
                if (pretend_to_fail) continue ;

                // perform solution check if user wish to do so
                bool Is_correct;
                TEST_CHECK(spex_check_solution(&Is_correct, A, sol, b, option));
                if (pretend_to_fail) continue ;
                OK (SPEX_matrix_free(&sol, option));
                TEST_ASSERT (sol == NULL);
                if (pretend_to_fail) continue ;

            }
            else
            {
                //--------------------------------------------------------------
                // SPEX Left LU backslash
                // solve Ax=b in double precision
                //--------------------------------------------------------------

                TEST_ASSERT (sol_doub == NULL);
                TEST_CHECK(SPEX_lu_backslash(&sol_doub, SPEX_FP64, A, b,
                    option));
                if (pretend_to_fail) continue ;
                OK (SPEX_matrix_free(&sol_doub, option));
                TEST_ASSERT (sol_doub == NULL);

                // failure case
                if (Ab_type == 1)
                {
                    TEST_CHECK_FAILURE(SPEX_lu_factorize(NULL,
                        A, NULL, NULL), SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;
                    // incorrect solution type
                    TEST_CHECK_FAILURE(SPEX_lu_backslash(&sol, SPEX_MPZ,
                        A, b, option), SPEX_INCORRECT_INPUT);
                    TEST_ASSERT (sol == NULL);
                    if (pretend_to_fail) continue ;
                    // NULL solution pointer
                    TEST_CHECK_FAILURE(SPEX_lu_backslash(NULL, SPEX_MPZ,
                        A, b, option), SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;
                    // invalid kind
                    A->kind = 4;
                    int64_t tmp;
                    TEST_CHECK_FAILURE(SPEX_matrix_nnz(&tmp, A, NULL),
                        SPEX_INCORRECT_INPUT);
                    if (pretend_to_fail) continue ;
                    A->kind = SPEX_CSC;
                }
            }

            //------------------------------------------------------------------
            // test SPEX_lu_analyze
            //------------------------------------------------------------------
            if (A != NULL && option != NULL)
            {
                SPEX_preorder saved_order = option->order;
                option->order = SPEX_NO_ORDERING;
                TEST_ASSERT (S == NULL);
                TEST_CHECK(SPEX_lu_analyze(&S, A, option));
                if (pretend_to_fail) continue ;
                if (option != NULL) option->order = saved_order;
            }

            //------------------------------------------------------------------
            // test spex_amd and spex_colamd
            //------------------------------------------------------------------

            int64_t *ignore = NULL, nnz = 0 ;

            if (A != NULL)
            {
                int64_t nsave = A->n ;
                A->n = -1 ;
                TEST_CHECK_FAILURE (spex_amd (&ignore, &nnz, A, NULL),
                    SPEX_INCORRECT_INPUT);
                if (pretend_to_fail) continue ;
                if (A != NULL) A->n = nsave ;
            }

            if (A != NULL)
            {
                A->p [0] = 1 ;
                TEST_CHECK_FAILURE (spex_colamd (&ignore, &nnz, A, NULL),
                    SPEX_INCORRECT_INPUT);
                if (pretend_to_fail) continue ;
                if (A != NULL) A->p [0] = 0 ;
            }

            //------------------------------------------------------------------
            // Free Memory
            //------------------------------------------------------------------

            SPEX_FREE_ALL;
            if(!is_simple_test)
            {
                if (malloc_count > 0)
                {
                    malloc_count_list[k] = kk;
                    break;
                }
                else {continue;}
            }
        }
    }

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    if (is_simple_test)
    {
        free(gmp_ntrial_list);
        free(malloc_trials_list);
        printf ("\nSPEX LU tests finished, tests passed\n");
    }
    else
    {
        printf("least required malloc_count for Ab_type = 0~20 are ");
        for (int i = 0; i < 20; i++)
        {
            printf("%ld ", malloc_count_list[i]);
        }
        printf("\nSPEX LU: brutal tests finished, tests passed\n");
    }

    printf ("%s: all tests passed\n\n", __FILE__);
    fprintf (stderr, "%s: all tests passed\n\n", __FILE__);
    return 0;
}

