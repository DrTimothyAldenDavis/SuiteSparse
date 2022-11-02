//------------------------------------------------------------------------------
// SLIP_LU/Tcov/tcov_test.c: test coverage for SLIP_LU
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

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
 * N and list1 specify the test list for slip_gmp_ntrials (in SLIP_gmp.h)
 * M and list2 specify the test list for malloc_count (in tcov_malloc_test.h)
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
 * uncomment to show the input lists for slip_gmp_ntrials and malloc_count
 */
// #define SLIP_TCOV_SHOW_LIST

/* This program will exactly solve the sparse linear system Ax = b by performing
 * the SLIP LU factorization. Refer to README.txt for information on how
 * to properly use this code.
 */

#define SLIP_FREE_ALL                            \
{                                                \
    SLIP_matrix_free(&A,option);                 \
    SLIP_matrix_free(&b, option);                \
    SLIP_matrix_free(&B, option);                \
    SLIP_matrix_free(&Ax, option);               \
    SLIP_matrix_free(&sol, option);              \
    SLIP_FREE(option);                           \
    SLIP_finalize() ;                            \
}

#include "tcov_malloc_test.h"

#define TEST_CHECK(method)                       \
{                                                \
    info = (method) ;                            \
    if (info != SLIP_OK)                         \
    {                                            \
        SLIP_PRINT_INFO (info) ;                 \
        SLIP_FREE_ALL;                           \
        continue;                                \
    }                                            \
}

#define TEST_CHECK_FAILURE(method)               \
{                                                \
    info = (method) ;                            \
    if (info != SLIP_INCORRECT_INPUT && info != SLIP_SINGULAR) \
    {                                            \
        SLIP_PRINT_INFO (info) ;                 \
        SLIP_FREE_ALL ;                          \
        continue ;                               \
    }                                            \
    else                                         \
    {                                            \
        printf("Expected failure at line %d\n", __LINE__);\
    }                                            \
}

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

#include <assert.h>

int main( int argc, char* argv[])
{
    bool IS_SIMPLE_TEST = true;
    int Ab_type = 0;
    int64_t malloc_count_list[20]= { -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1};
    int64_t NUM_OF_TRIALS = 0 ;
    int64_t NUM_OF_MALLOC_T = 0;
    int64_t *gmp_ntrial_list=NULL;         // only used in simple test
    int64_t *malloc_trials_list=NULL;          // only used in simple test

    //--------------------------------------------------------------------------
    // parse input arguments
    //--------------------------------------------------------------------------

    if (argc == 1)                         // brutal test
    {
        IS_SIMPLE_TEST = false;
        NUM_OF_TRIALS = 20;
    }
    else                                   // simple test
    {
        IS_SIMPLE_TEST = true;

        int64_t arg_count = 0;
        // type of Matrix A and vector b:
        // 0 mpz, 1 double, 2 int64_t, 3 mpq, 4 mpfr
        Ab_type = atoi(argv[++arg_count]);
        if (!argv[++arg_count])
        {
            NUM_OF_TRIALS=1;
            gmp_ntrial_list= malloc (NUM_OF_TRIALS* sizeof(int64_t));
            gmp_ntrial_list[0]=-1;
            arg_count--;
        }
        else
        {
            NUM_OF_TRIALS=atoi(argv[arg_count]);
            gmp_ntrial_list= malloc (NUM_OF_TRIALS* sizeof(int64_t));
            for (int64_t k=0; k<NUM_OF_TRIALS; k++)
            {
                if (argv[++arg_count])
                {
                    gmp_ntrial_list[k]=atoi(argv[arg_count]);
                }
                else
                {
                    fprintf(stderr, "WARNING: MISSING gmp trial\n");
                    NUM_OF_TRIALS=1;
                    gmp_ntrial_list[0]=-1;
                    arg_count--;
                }
            }
        }
        if (!argv[++arg_count])
        {
            NUM_OF_MALLOC_T=1;
            malloc_trials_list= malloc (NUM_OF_MALLOC_T* sizeof(int64_t));
            malloc_trials_list[0]=MAX_MALLOC_COUNT;//INT_MAX;
        }
        else
        {
            NUM_OF_MALLOC_T=atoi(argv[arg_count]);
            malloc_trials_list= malloc (NUM_OF_MALLOC_T* sizeof(int64_t));
            for (int64_t k=0; k<NUM_OF_MALLOC_T; k++)
            {
                if (argv[++arg_count])
                {
                    malloc_trials_list[k]=atoi(argv[arg_count]);
                }
                else
                {
                    fprintf(stderr, "WARNING: MISSING malloc trial\n");
                    NUM_OF_MALLOC_T=1;
                    malloc_trials_list[0]=MAX_MALLOC_COUNT;//INT_MAX;
                }
            }
        }

        #ifdef SLIP_TCOV_SHOW_LIST
        printf ("gmp ntrials list is: ");
        for (int64_t k=0; k<NUM_OF_TRIALS; k++)
        {
            printf("%ld   ",gmp_ntrial_list[k]);
        }
        printf("\nmalloc trial list is: ");
        for (int64_t k=0; k<NUM_OF_MALLOC_T; k++)
        {
            printf("%d   ",malloc_trials_list[k]);
        }
        printf("\n");
        #endif /* SLIP_TCOV_SHOW_LIST */
    }

    //--------------------------------------------------------------------------
    // test calloc, realloc, free
    //--------------------------------------------------------------------------

    SLIP_info info ;
    info = SLIP_initialize ( ) ;                       assert (info == SLIP_OK) ;
    info = SLIP_finalize ( ) ;                         assert (info == SLIP_OK) ;
    info = SLIP_initialize ( ) ;                       assert (info == SLIP_OK) ;
    int *p4 = SLIP_calloc (5, sizeof (int)) ;          assert (p4 != NULL)  ;
    bool ok ;
    p4 = SLIP_realloc (6, 5, sizeof (int), p4, &ok) ;  assert (ok) ;
    info = SLIP_finalize ( ) ;                         assert (info == SLIP_OK) ;
    p4 = SLIP_realloc (7, 6, sizeof (int), p4, &ok) ;  assert (!ok) ;
    info = SLIP_initialize ( ) ;                       assert (info == SLIP_OK) ;
    SLIP_FREE (p4) ;
    info = SLIP_finalize ( ) ;                         assert (info == SLIP_OK) ;

    //--------------------------------------------------------------------------
    // run all trials
    //--------------------------------------------------------------------------

    // For SIMPLE_TEST, outer loop iterates for slip_gmp_ntrials initialized
    // from list1 (input for tcov_test) and inner loop interates for
    // malloc_count initialized from list2 (input for tcov_test).
    //
    // For non SIMPLE_TEST, outer loop iterates for Ab_type from 0 to 5, and
    // inner loop iterates for malloc_count initialized from 0 to
    // MAX_MALLOC_COUNT, break when malloc_count>0 at the end of inner loop.

    for (int64_t k=0; k<NUM_OF_TRIALS; k++)
    {
        if (IS_SIMPLE_TEST)
        {
            // only the first outter loop will iterate across all list2
            if (k == 1)
            {
                NUM_OF_MALLOC_T=1;
                malloc_trials_list[0]=INT_MAX;
            }
        }
        else
        {
            Ab_type = k;
            NUM_OF_MALLOC_T = MAX_MALLOC_COUNT;
        }

        for (int64_t kk=0; kk<NUM_OF_MALLOC_T; kk++)
        {
            if (IS_SIMPLE_TEST)
            {
                slip_gmp_ntrials=gmp_ntrial_list[k];
                printf("initial slip_gmp_ntrials=%ld\n",slip_gmp_ntrials);
                malloc_count=malloc_trials_list[kk];
                printf("%"PRId64" out of %"PRId64", "
                    "initial malloc_count=%"PRId64"\n",
                    kk, NUM_OF_MALLOC_T, malloc_count);
            }
            else
            {
                malloc_count = kk;
                printf("[Ab_type malloc_count] = [%d %"PRId64"]\n",
                    Ab_type, malloc_count);
            }

            //------------------------------------------------------------------
            // Initialize SLIP LU process
            //------------------------------------------------------------------

            SLIP_initialize_expert (tcov_malloc, tcov_calloc,
                tcov_realloc, tcov_free) ;

            info = SLIP_initialize ( ) ;
            assert (info == SLIP_PANIC) ;

            //------------------------------------------------------------------
            // Allocate memory
            //------------------------------------------------------------------

            int64_t n=4, numRHS=1, j, nz=11;
            SLIP_options* option = SLIP_create_default_options();
            if (!option) {continue;}
            option->print_level = 3;

            // used in as source in different Ab_type for A and b
            SLIP_matrix *B   = NULL;
            SLIP_matrix *Ax  = NULL;

            // matrix A, b and solution
            SLIP_matrix *A = NULL ;
            SLIP_matrix *b = NULL ;
            SLIP_matrix *sol = NULL;

            if (Ab_type >= 0 && Ab_type <= 4)
            {

                //--------------------------------------------------------------
                // Solve A*x=b where A and b are created from mpz entries
                //--------------------------------------------------------------

                TEST_CHECK(SLIP_matrix_allocate(&B, SLIP_DENSE,
                    (SLIP_type) Ab_type, n,
                    numRHS, n*numRHS, false, true, option));
                TEST_CHECK(SLIP_matrix_allocate(&Ax, SLIP_CSC,
                    (SLIP_type) Ab_type, n,
                    n, nz, false, true, option));

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
                    TEST_CHECK(SLIP_matrix_copy(&A, SLIP_CSC, SLIP_MPZ, Ax,
                        option));
                    TEST_CHECK(SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ, B,
                        option));
                    // to trigger SLIP_SINGULAR 
                    TEST_CHECK_FAILURE(SLIP_backslash(&sol, SLIP_MPQ, A, b,
                        option));
                    option->pivot = SLIP_LARGEST;
                    TEST_CHECK_FAILURE(SLIP_backslash(&sol, SLIP_MPQ, A, b,
                        option));
                    option->pivot = SLIP_FIRST_NONZERO;
                    TEST_CHECK_FAILURE(SLIP_backslash(&sol, SLIP_MPQ, A, b,
                       option));

                    //free the memory alloc'd
                    SLIP_matrix_free (&A, option) ;
                    SLIP_matrix_free (&b, option) ;

                    // trigger gcd == 1
                    int32_t prec = option->prec;
                    option->prec = 17;
                    double pow2_17 = pow(2,17);
                    for (j = 0; j < n; j++)                             // Get B
                    {
                        TEST_CHECK(SLIP_mpfr_set_d(SLIP_2D(B,j,0,mpfr),
                            bxnum[j]/pow2_17, MPFR_RNDN));
                    }
                    TEST_CHECK(SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ,B,
                        option));
                    SLIP_matrix_free (&b, option) ;

                    // restore default precision
                    option->prec = prec;

                    // use diagonal entries as pivot
                    option->pivot = SLIP_DIAGONAL;
                }
                else if (Ab_type == 4)// double
                {
                    // create empty A using uninitialized double mat/array
                    // to trigger all-zero array condition
                    TEST_CHECK(SLIP_matrix_copy(&A, SLIP_CSC, SLIP_MPZ, Ax,
                        option));
                    SLIP_matrix_free (&A, option) ;

                    // trigger gcd == 1
                    for (j = 0; j < n; j++)                           // Get b
                    {
                        SLIP_2D(B,j,0,fp64) = bxnum[j]/1e17;
                    }
                    TEST_CHECK(SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ, B,
                        option));
                    SLIP_matrix_free (&b, option) ;

                    // use smallest entry as pivot
                    option->pivot = SLIP_SMALLEST;
                }

                // fill Ax->x and b->x
                for (j = 0; j < n; j++)                           // Get b
                {
                    if (Ab_type == 0) //MPZ
                    {
                        TEST_CHECK(SLIP_mpz_set_ui(SLIP_2D(B, j, 0, mpz),
                            bxnum3[j]));
                    }
                    else if (Ab_type == 1)// MPQ
                    {
                        TEST_CHECK(SLIP_mpq_set_ui(SLIP_2D(B,j,0,mpq),
                            bxnum3[j], bxden3[j]));
                    }
                    else if (Ab_type == 2)// MPFR
                    {
                        TEST_CHECK(SLIP_mpfr_set_d(SLIP_2D(B,j,0,mpfr),bxnum[j],
                            MPFR_RNDN));
                        TEST_CHECK(SLIP_mpfr_div_d(SLIP_2D(B,j,0,mpfr),
                            SLIP_2D(B,j,0,mpfr), bxden[j], MPFR_RNDN));
                    }
                    else if (Ab_type == 3)// INT64
                    {
                        SLIP_2D(B,j,0,int64)=bxnum3[j];
                    }
                    else // double
                    {
                        SLIP_2D(B,j,0,fp64) = bxnum[j];
                    }
                }
                for (j = 0; j < nz; j++)                          // Get Ax
                {
                    if (Ab_type == 0)
                    {
                        TEST_CHECK(SLIP_mpz_set_ui(Ax->x.mpz[j],Axnum3[j]));
                    }
                    else if (Ab_type == 1)
                    {
                        TEST_CHECK(SLIP_mpq_set_ui(Ax->x.mpq[j],Axnum3[j],
                            Axden3[j]));
                    }
                    else if (Ab_type == 2)
                    {
                        TEST_CHECK(SLIP_mpfr_set_d(Ax->x.mpfr[j], Axnum[j],
                            MPFR_RNDN));
                        TEST_CHECK(SLIP_mpfr_div_d(Ax->x.mpfr[j], Ax->x.mpfr[j],
                            Axden[j], MPFR_RNDN))
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

                // successful case
                TEST_CHECK(SLIP_matrix_copy(&A, SLIP_CSC, SLIP_MPZ, Ax,option));
                TEST_CHECK(SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ,B,option));
                SLIP_matrix_free(&B, option);
                SLIP_matrix_free(&Ax, option);
            }
            else // 5 =< Ab_type < 20
            {

                //--------------------------------------------------------------
                // Test SLIP_matrix_copy and SLIP_matrix_check brutally
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
                TEST_CHECK(SLIP_matrix_allocate(&Ax, (SLIP_type) kind,
                    (SLIP_type)type, m1, n1, nz1, false, true, option));
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
                for (j = 0; j < nz; j++)
                {
                    if (kind != 2) {Ax->i[j] = I[j];}
                    // triplet
                    if (kind == 1){  Ax->j[j] = J[j];}
                    switch (type)
                    {
                        case 0: // MPZ
                        {
                            TEST_CHECK(SLIP_mpz_set_si(Ax->x.mpz[j],
                                x_int64[j]));
                        }
                        break;

                        case 1: // MPQ
                        {
                            TEST_CHECK(SLIP_mpq_set_ui(Ax->x.mpq[j],
                                2*x_int64[j],2));
                        }
                        break;

                        case 2: // MPFR
                        {
                            TEST_CHECK(SLIP_mpfr_set_d(Ax->x.mpfr[j],
                                x_doub2[j], MPFR_RNDN));
                            TEST_CHECK(SLIP_mpfr_div_d(Ax->x.mpfr[j],
                                Ax->x.mpfr[j], 1, MPFR_RNDN));
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
                }
                TEST_CHECK (SLIP_matrix_check (Ax, option));

                // convert to all different type of matrix
                for (int tk1 = 0; tk1 < 15 && info == SLIP_OK; tk1++)
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
                        ? "int64" : "double",type1) ;

                    TEST_CHECK(SLIP_matrix_copy(&A, (SLIP_kind)kind1,
                        (SLIP_type) type1, Ax, option));

                    TEST_CHECK (SLIP_matrix_check (A, NULL));

                    // just perform once to try some failure cases
                    if (tk == 0 && tk1 == 0)
                    {
                        // fail SLIP_LU_solve
                        TEST_CHECK(SLIP_matrix_allocate(&b, SLIP_DENSE,
                            SLIP_MPZ, 1, 1, 1, true, true, option));
                        TEST_CHECK_FAILURE(SLIP_LU_solve(NULL, b, A, A, A,
                            b, NULL, NULL, option));
                        SLIP_matrix_free (&b, option) ;
                    }
                    else if (tk == 0 && (tk1 == 1 || tk1 == 2 || tk1 == 4))
                    {
                        // test SLIP_matrix_copy with scale
                        SLIP_matrix_free (&A, option) ;
                        TEST_CHECK (SLIP_mpq_set_ui (Ax->scale, 2, 5)) ;
                        TEST_CHECK(SLIP_matrix_copy(&A, (SLIP_kind)kind1,
                            (SLIP_type) type1, Ax, option));
                        TEST_CHECK (SLIP_mpq_set_ui (Ax->scale, 1, 1)) ;
                    }
                    else if (tk == 0 && tk1 == 3)//A = CSC int64
                    {
                        // test SLIP_matrix_check
                        A->i[0] = -1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        SLIP_FREE(A->x.int64);
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        A->p[1] = 2;
                        A->p[2] = 1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        A->p[0] = 1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        A->type = -1;// invalid type
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        A->nzmax = -1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        A->m = -1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        TEST_CHECK_FAILURE(SLIP_matrix_check(NULL, option));

                        // Incorrect calling with NULL pointer(s)
                        TEST_CHECK_FAILURE(SLIP_LU_analyze(NULL,A,NULL));

                        // test SLIP_matrix_copy with scale
                        SLIP_matrix_free (&A, option) ;
                        TEST_CHECK (SLIP_mpq_set_ui (Ax->scale, 5, 2)) ;
                        TEST_CHECK(SLIP_matrix_copy(&A, (SLIP_kind)kind1,
                            (SLIP_type) type1, Ax, option));
                        TEST_CHECK (SLIP_mpq_set_ui (Ax->scale, 1, 1)) ;
                    }
                    else if (tk == 0 && tk1 == 8) // A= Triplet int64
                    {
                        // test SLIP_matrix_check
                        A->i[0] = -1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        SLIP_FREE(A->x.int64);
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                        A->n = -1;
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                    }
                    else if (tk == 0 && tk1 == 13)//A= dense int64
                    {
                        SLIP_FREE(A->x.int64);
                        TEST_CHECK_FAILURE(SLIP_matrix_check(A, option));
                    }
                    SLIP_matrix_free (&A, option) ;
                    info = SLIP_OK;

                }
                TEST_CHECK(info);
                if (tk == 3)
                {
                    // fail SLIP_matrix_copy
                    TEST_CHECK_FAILURE(SLIP_matrix_copy(&A, 7,
                        (SLIP_type) type, Ax, option));
                    // failure case: Ax->x = NULL
                    SLIP_FREE(Ax->x.int64);
                    TEST_CHECK_FAILURE(SLIP_matrix_copy(&A, SLIP_CSC, SLIP_MPZ,
                        Ax,option));

                    // fail SLIP_matrix_allocate
                    TEST_CHECK_FAILURE(SLIP_matrix_allocate(NULL,
                        SLIP_DENSE, SLIP_MPZ, 1, 1, 1,
                        true, true, option));
                    TEST_CHECK_FAILURE(SLIP_matrix_allocate(&b,
                        SLIP_DENSE, SLIP_MPZ, -1, 1, 1,
                        true, true, option));

                    // test SLIP_matrix_allocate
                    TEST_CHECK(SLIP_matrix_allocate(&b, SLIP_DENSE,
                        SLIP_MPQ, 1, 1, 1, false, false, option));
                    SLIP_matrix_free (&b, option) ;
                    TEST_CHECK(SLIP_matrix_allocate(&b, SLIP_DENSE,
                        SLIP_MPFR, 1, 1, 1, false, false, option));
                    SLIP_matrix_free (&b, option) ;

                    //test coverage for slip_gmp_reallocate()
                    void *p_new = NULL;
                    TEST_CHECK(slip_gmp_realloc_test(&p_new, NULL,0,1));
                    TEST_CHECK(slip_gmp_realloc_test(&p_new,p_new,1,0));
                }

                //--------------------------------------------------------------
                // test SLIP_matrix_check on a triplet matrix with bad triplets
                //--------------------------------------------------------------

                printf ("\n[ SLIP_matrix_check -------------------------\n") ;
                SLIP_matrix_free (&A, option) ;
                int64_t I2 [4] = { 1, 2, 1, 1 } ;
                int64_t J2 [4] = { 1, 0, 0, 1 } ;
                TEST_CHECK (SLIP_matrix_allocate (&A, SLIP_TRIPLET,
                    SLIP_INT64, 3, 3, 4, true, false, option)) ;
                A->i = I2 ;
                A->j = J2 ;
                A->x.int64 = I2 ;
                A->nz = 4 ;
                printf ("invalid triplet matrix expected:\n") ;
                TEST_CHECK_FAILURE (SLIP_matrix_check (A, option)) ;
                SLIP_matrix_free (&A, option) ;

                TEST_CHECK (SLIP_matrix_allocate (&A, SLIP_CSC,
                    SLIP_INT64, 3, 3, 4, true, false, option)) ;
                int64_t P3 [4] = { 0, 2, 4, 4 } ;
                int64_t I3 [4] = { 0, 0, 0, 0 } ;
                A->p = P3 ;
                A->i = I3 ;
                A->x.int64 = I3 ;
                printf ("invalid CSC matrix expected:\n") ;
                TEST_CHECK_FAILURE (SLIP_matrix_check (A, option)) ;
                SLIP_matrix_free (&A, option) ;
                printf ("-----------------------------------------------]\n") ;

                //--------------------------------------------------------------

                SLIP_FREE_ALL;

                // for miscellaneous test, continue to next loop directly
                if (!IS_SIMPLE_TEST)
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
                // SLIP LU backslash
                // solve Ax=b in full precision rational arithmetic
                //--------------------------------------------------------------
		TEST_CHECK(SLIP_backslash(&sol, SLIP_MPQ, A, b, option));

		if (Ab_type == 4)
                {
                    // This would return SLIP_INCORRECT since sol has been
                    // scaled down so that sol->scale = 1. Therefore sol is
                    // solution for original unscaled Ax=b, while this is
                    // checking if x is the solution for scaled Ax=b
                    info = slip_check_solution(A, sol, b, option);
		    if (info == SLIP_INCORRECT) {;}
                    else {TEST_CHECK(info);}
                }

            }
            else
            {
                //--------------------------------------------------------------
                // SLIP LU backslash
                // solve Ax=b in double precision
                //--------------------------------------------------------------
                SLIP_matrix *sol_doub;
		TEST_CHECK(SLIP_backslash(&sol_doub, SLIP_FP64, A, b, option));
                SLIP_matrix_free(&sol_doub, option);

                // failure case
                if (Ab_type == 1)
                {
                    TEST_CHECK_FAILURE(SLIP_LU_factorize(NULL, NULL,
                        NULL, NULL, A, NULL, NULL));
                    // incorrect solution type
                    TEST_CHECK_FAILURE(SLIP_backslash(&sol, SLIP_MPZ, A, b,
                       option));
                    // NULL solution pointer
                    TEST_CHECK_FAILURE(SLIP_backslash(NULL, SLIP_MPZ, A, b,
                       option));
                    // invalid kind
                    A->kind = 4;
                    SLIP_matrix_nnz(A, NULL);
                    A->kind = SLIP_CSC;
                }

            }

            //------------------------------------------------------------------
            // Free Memory
            //------------------------------------------------------------------

            SLIP_FREE_ALL;
            if(!IS_SIMPLE_TEST)
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

    if (IS_SIMPLE_TEST)
    {
        free(gmp_ntrial_list);
        free(malloc_trials_list);
        printf ("tests finished\n") ;
    }
    else
    {
        printf("least required malloc_count for Ab_type = 0~20 are ");
        for (int i = 0; i < 20; i++)
        {
            printf("%ld ", malloc_count_list[i]);
        }
        printf("\nbrutal tests finished\n");
    }

    return 0;
}

