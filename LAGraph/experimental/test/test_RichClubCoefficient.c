//----------------------------------------------------------------------------
// LAGraph/src/test/test_RichClubCoefficient.c: test cases for 
// LAGraph_RichClubCoefficient
//----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

//-----------------------------------------------------------------------------

// This program tests Rich Club Coefficient by comparing it to know values.

#include <stdio.h>
#include <acutest.h>
#include <LAGraphX.h>
#include <LAGraph_test.h>
#include <LG_Xtest.h>
#include <LG_test.h>

char msg [LAGRAPH_MSG_LEN] ;

GrB_Matrix A = NULL, C = NULL;
GrB_Vector rcc = NULL, check_rcc = NULL ;
LAGraph_Graph G = NULL ;

#define LEN 512
char filename [LEN+1] ;
typedef struct
{
    double *rcc ; // for unweighted matchings, the size of the matching. For weighted, sum of edge weights
    uint64_t n;
    const char *name ;    // matrix file name
}
matrix_info ;

double rcc1[] = {0.08489795918367347, 0.09042553191489362, 0.11806543385490754,
0.15270935960591134, 0.17543859649122806, 0.18681318681318682,
0.3333333333333333, 0.3333333333333333};

double rcc2[] = {0.048040201005025124, 0.048040201005025124,
0.04842393787117405, 0.04945054945054945, 0.05129490392648287,
0.052702702702702706, 0.05523809523809524, 0.0613164184592756,
0.06755555555555555, 0.07603960396039604, 0.08637747336377473,
0.09183673469387756, 0.09090909090909091, 0.11578947368421053,
0.15555555555555556, 0.16666666666666666, 0.0};

double rcc3[] = {0.020418922066450775, 0.020418922066450775,
0.020418922066450775, 0.020418922066450775, 0.020683173955750592,
0.02150664338158559, 0.02150664338158559, 0.02150664338158559,
0.02150664338158559, 0.021543516982986302, 0.02175414105479627,
0.02177185436416472, 0.0218052051134787, 0.021857560922981484,
0.02201878369318151, 0.022188402920223206, 0.022804582640104137,
0.02498473901586159, 0.025249731948717845, 0.02596385205080857,
0.027247042660294644, 0.027751420992800303, 0.028748882924051613,
0.030170594138205473, 0.031102722135686933, 0.03269071555292726,
0.03991334958028703, 0.04169452474008947, 0.04259653806582863,
0.044304609453855226, 0.04526841567726286, 0.04650692548781721,
0.049532888465204955, 0.05002586270597798, 0.05063092496587295,
0.05300441583096034, 0.054629398879867785, 0.05653826181371855,
0.059291297964366496, 0.06080825884426539, 0.06422637670884515,
0.06768885564697083, 0.06889041561605802, 0.0701751819478477,
0.07607985994153141, 0.07831153079788616, 0.07894806048652203,
0.08038113301271196, 0.08207037795568796, 0.0836966039974926,
0.08430976691170078, 0.08589112981595826, 0.08725832508207827,
0.10235720633666719, 0.10276089969086451, 0.10358862936809705,
0.10741510741510742, 0.1110049401354954, 0.11278770872986284,
0.11678584477269169, 0.11817043607981033, 0.12116372726010276,
0.1218450408924313, 0.12440239209741634, 0.12798272259582463,
0.13977635782747605, 0.14573643410852713, 0.14652869744786873,
0.1486150774302895, 0.15837726930369062, 0.158866930171278,
0.16537043438184124, 0.16662279547249276, 0.1688366106970758,
0.17170333945410973, 0.17634190936398067, 0.17711727724564694,
0.17787300615583443, 0.1779243342906783, 0.17813492063492065,
0.16556371804990588, 0.16556371804990588, 0.16556371804990588,
0.16556371804990588, 0.16729064039408867, 0.1701346389228886,
0.1989280560709132, 0.20149602618045817, 0.20475808607324245,
0.16666666666666666, 0.16840882694541232, 0.17894736842105263,
0.16666666666666666, 1.0} ;
double rcc4[] = {0.0016506547800326698, 0.0017226315730560155, 
    0.0034201182512489416, 0.0037033309852068028, 0.05405405405405406};
const matrix_info tests [ ] =
{
    {rcc1, sizeof(rcc1) / sizeof(rcc1[0]), "random_unweighted_general1.mtx"},
    {rcc2, sizeof(rcc2) / sizeof(rcc2[0]), "random_unweighted_general2.mtx"},
    {rcc3, sizeof(rcc3) / sizeof(rcc3[0]), "bcsstk13.mtx"},
    {rcc4, sizeof(rcc4) / sizeof(rcc4[0]), "test_FW_2500.mtx"},
    {NULL, 0, ""}
} ;

const char *tests2 [ ] =
{
    "random_unweighted_general1.mtx",
    "random_unweighted_general2.mtx",
    "bcsstk13.mtx",
    "bcsstk13_celeb.mtx",
    "test_FW_1000.mtx",
    "test_FW_2003.mtx",
    "test_FW_2500.mtx",
    ""
} ;

#undef OK
#define OK(method) \
{ \
    GrB_Info info = method ; \
    if (info != GrB_SUCCESS) \
    { \
        printf ("FAIL at %s, %d: info: %d, msg: %s\n", __FILE__, \
            __LINE__, info, msg) ; \
        TEST_CHECK (false) ; \
    } \
}

void test_RichClubCoefficient (void)
{
    //--------------------------------------------------------------------------
    // start LAGraph
    //--------------------------------------------------------------------------
    OK (LAGraph_Init (msg)) ;
    

    for (int k = 0 ; ; k++)
    {
        //The following code taken from MIS tester
        // load the matrix as A
        const char *aname = tests [k].name;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;
        printf ("\nMatrix: %s\n", aname) ;
        const double *ans = tests [k].rcc;
        const uint64_t n_ans = tests [k].n;

        // C = structure of A
        OK (LAGraph_Matrix_Structure (&C, A, msg)) ;
        OK (GrB_free (&A)) ;

        // construct a directed graph G with adjacency matrix C
        OK (LAGraph_New (&G, &C, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (C == NULL) ;

        // check if the pattern is symmetric
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            // make the adjacency matrix symmetric
            OK (LAGraph_Cached_AT (G, msg)) ;
            OK (GrB_eWiseAdd (G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL)) ;
            G->is_symmetric_structure = LAGraph_TRUE ;
        }
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;

        // check for self-edges
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }

        // compute the row degree
        OK (LAGraph_Cached_OutDegree (G, msg)) ;

        //----------------------------------------------------------------------
        // test the algorithm
        //----------------------------------------------------------------------

        printf ("RCC computation begins:\n") ;
        GrB_set (GrB_GLOBAL, (int32_t) (true), GxB_BURBLE) ;
        OK(LAGraph_RichClubCoefficient ( &rcc, G, msg));
        printf("%s\n", msg);
        GrB_set (GrB_GLOBAL, (int32_t) (false), GxB_BURBLE) ;
        printf ("RCC computation ends:\n") ;

        //----------------------------------------------------------------------
        // check results
        //----------------------------------------------------------------------
        double comp_val = 0;
        for(int64_t i = n_ans - 1; i >= 0; --i)
        {
            GrB_Vector_extractElement(&comp_val, rcc, i) ;
            TEST_CHECK (
                comp_val - ans[i] <= 1e-10 && ans[i] - comp_val <= 1e-10) ;
        }
        GxB_Vector_fprint (rcc, "rcc", GxB_SHORT, stdout);
        OK (GrB_free (&rcc)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------
    OK (LAGraph_Finalize (msg)) ;
}

void iseq(bool *z, const double *x, const double *y)
{
    (*z) = (isnan(*x) && isnan(*y)) ||*x == *y ;
}
//------------------------------------------------------------------------------
// test RichClubCoefficient vs C code
//------------------------------------------------------------------------------
void test_RCC_Check (void)
{
    //--------------------------------------------------------------------------
    // start LAGraph
    //--------------------------------------------------------------------------
    OK (LAGraph_Init (msg)) ;
    GrB_BinaryOp iseqFP = NULL ;
    OK (GrB_BinaryOp_new (
        &iseqFP, (GxB_binary_function) iseq, GrB_BOOL, GrB_FP64, GrB_FP64)) ;
    // OK (GxB_BinaryOp_new (
    //     &iseqFP, (GxB_binary_function) iseq, 
    //     GrB_BOOL, GrB_FP64, GrB_FP64, "iseq", ISEQ)) ;
    for (int k = 0 ; ; k++)
    {
        //The following code taken from MIS tester
        // load the matrix as A
        const char *aname = tests2 [k];
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;
        printf ("\nMatrix: %s\n", aname) ;

        // C = structure of A
        OK (LAGraph_Matrix_Structure (&C, A, msg)) ;
        OK (GrB_free (&A)) ;

        // construct a directed graph G with adjacency matrix C
        OK (LAGraph_New (&G, &C, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (C == NULL) ;

        // check if the pattern is symmetric
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            // make the adjacency matrix symmetric
            OK (LAGraph_Cached_AT (G, msg)) ;
            OK (GrB_eWiseAdd (G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL)) ;
            G->is_symmetric_structure = LAGraph_TRUE ;
        }
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;

        // check for self-edges
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }

        // compute the row degree
        OK (LAGraph_Cached_OutDegree (G, msg)) ;

        //----------------------------------------------------------------------
        // test the algorithm
        //----------------------------------------------------------------------

        // GrB_set (GrB_GLOBAL, (int32_t) (true), GxB_BURBLE) ;
        OK(LAGraph_RichClubCoefficient ( &rcc, G, msg));
        // GrB_set (GrB_GLOBAL, (int32_t) (false), GxB_BURBLE) ;

        OK(LG_check_rcc(&check_rcc, G, msg));
        //----------------------------------------------------------------------
        // check results
        //----------------------------------------------------------------------
        bool flag = false;
        OK (LAGraph_Vector_IsEqualOp(&flag, rcc, check_rcc, iseqFP, msg)) ;
        TEST_CHECK (flag) ;
        GxB_Vector_fprint (rcc, "rcc", GxB_SHORT, stdout);
        GxB_Vector_fprint (check_rcc, "check_rcc", GxB_SHORT, stdout);
        OK (GrB_free (&rcc)) ;
        OK (GrB_free (&check_rcc)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------
    OK (GrB_free (&iseqFP)) ;
    OK (LAGraph_Finalize (msg)) ;
}
//------------------------------------------------------------------------------
// test_RCC_brutal:
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_rcc_brutal (void)
{
    //--------------------------------------------------------------------------
    // start LAGraph
    //--------------------------------------------------------------------------
    OK (LG_brutal_setup (msg)) ;
    

    for (int k = 0 ; ; k++)
    {
        // load the matrix as A
        const char *aname = tests [k].name;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;
        printf ("\nMatrix: %s\n", aname) ;
        const double *ans = tests [k].rcc;
        const uint64_t n_ans = tests [k].n;

        // C = structure of A
        OK (LAGraph_Matrix_Structure (&C, A, msg)) ;
        OK (GrB_free (&A)) ;

        // construct a directed graph G with adjacency matrix C
        OK (LAGraph_New (&G, &C, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (C == NULL) ;

        // check if the pattern is symmetric
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            // make the adjacency matrix symmetric
            OK (LAGraph_Cached_AT (G, msg)) ;
            OK (GrB_eWiseAdd (G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL)) ;
            G->is_symmetric_structure = LAGraph_TRUE ;
        }
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;

        // check for self-edges
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }

        // compute the row degree
        OK (LAGraph_Cached_OutDegree (G, msg)) ;

        LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;
        //----------------------------------------------------------------------
        // test the algorithm
        //----------------------------------------------------------------------

        printf ("RCC computation begins:\n") ;
        // GrB_set (GrB_GLOBAL, (int32_t) (true), GxB_BURBLE) ;
        LG_BRUTAL_BURBLE (LAGraph_RichClubCoefficient( &rcc, G, msg));
        printf("%s\n", msg);
        // GrB_set (GrB_GLOBAL, (int32_t) (false), GxB_BURBLE) ;
        printf ("RCC computation ends:\n") ;

        //----------------------------------------------------------------------
        // check results
        //----------------------------------------------------------------------
        double comp_val = 0;
        for(int64_t i = n_ans - 1; i >= 0; --i)
        {
            OK (GrB_Vector_extractElement(&comp_val, rcc, i)) ;
            TEST_CHECK (fabs(comp_val - ans[i]) <= 1e-15) ;
        }
        OK (GrB_free (&rcc)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------
    OK(LG_brutal_teardown(msg)) ;
}
#endif

//----------------------------------------------------------------------------
// the make program is created by acutest, and it runs a list of tests:
//----------------------------------------------------------------------------

TEST_LIST =
{
    {"RichClubCoefficient", test_RichClubCoefficient},
    #if LG_SUITESPARSE_GRAPHBLAS_V10
    {"RichClubCoefficient_Check", test_RCC_Check},
    #endif
    #if LAGRAPH_SUITESPARSE
    {"rcc_brutal", test_rcc_brutal},
    #endif
    {NULL, NULL}
} ;
