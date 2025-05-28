//----------------------------------------------------------------------------
// LAGraph/src/test/test_SwapEdges.c: test cases for LAGraph_HelloWorld
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



#include <stdio.h>
#include <acutest.h>
#include <LAGraphX.h>
#include <LG_internal.h>
#include <LAGraph_test.h>
#include <LG_Xtest.h>
#include <LG_test.h>

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL, G_new = NULL;
GrB_Matrix A = NULL, C = NULL, C_new = NULL; 
#define LEN 512
char filename [LEN+1] ;

const char* tests [ ] =
{
    "random_unweighted_general1.mtx",
    "random_unweighted_general2.mtx",
    "random_weighted_general1.mtx",
    "random_weighted_general2.mtx",
    "bcsstk13.mtx",
    "test_FW_2500.mtx",
    ""
} ;
const char* testsb [ ] =
{
    "random_unweighted_general1.mtx",
    "random_unweighted_general2.mtx",
    "random_weighted_general1.mtx",
    "random_weighted_general2.mtx",
    "bucky.mtx",
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

void test_SwapEdges (void)
{
    #if LG_SUITESPARSE_GRAPHBLAS_V10
    //--------------------------------------------------------------------------
    // start LAGraph
    //--------------------------------------------------------------------------
    OK (LAGraph_Init (msg)) ;

    for (int k = 0 ; ; k++)
    {
        //The following code taken from MIS tester
        // load the matrix as A
        const char *aname = tests [k];
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
        GrB_Index n = 0;
        OK (LAGraph_Cached_OutDegree (G, msg)) ;
        OK (GrB_Matrix_nrows(&n, G->A));
        for (int jit = 0 ; jit <= 1 ; jit++)
        {
            OK (GxB_Global_Option_set (GxB_JIT_C_CONTROL,
                jit ? GxB_JIT_ON : GxB_JIT_OFF)) ;
            double pQ = 0.0;
            //------------------------------------------------------------------
            // test the algorithm
            //------------------------------------------------------------------
            // GrB_set (GrB_GLOBAL, (int32_t) (true), GxB_BURBLE) ;
            OK(LAGraph_SwapEdges( &G_new, &pQ, G, 100.0, msg));
            // GrB_set (GrB_GLOBAL, (int32_t) (false), GxB_BURBLE) ;
            printf ("Test ends. Swaps per Edge: %g \n", pQ) ;
            printf ("%s\n", msg) ;

            //------------------------------------------------------------------
            // check results
            //------------------------------------------------------------------
            //Make sure we got a symetric back out:
            OK (LAGraph_CheckGraph (G_new, msg)) ;
                    
            OK (LAGraph_Cached_NSelfEdges (G_new, msg)) ;
            TEST_CHECK (G_new->nself_edges == 0);
                    
            //Make sure no self edges created.
            OK (LAGraph_Cached_NSelfEdges (G_new, msg)) ;
            TEST_CHECK (G_new->nself_edges == 0);

            // Check nvals stay the same. 
            GrB_Index edge_count, new_edge_count;
            OK (GrB_Matrix_nvals(&edge_count, G->A)) ;
            OK (GrB_Matrix_nvals(&new_edge_count, G_new->A)) ;
            TEST_CHECK(edge_count == new_edge_count);
            //next: check degrees stay the same.
            OK (LAGraph_Cached_OutDegree (G_new, msg)) ;

            bool ok = false;
            OK (LAGraph_Vector_IsEqual (
                &ok, G->out_degree, G_new->out_degree, msg)) ;
            TEST_CHECK (ok) ;
            OK (LAGraph_Delete (&G_new, msg)) ;
        }
        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------
    LAGraph_Finalize (msg) ;
    #endif
}

void test_SwapFull(void)
{
    #if LG_SUITESPARSE_GRAPHBLAS_V10
    //--------------------------------------------------------------------------
    // start LAGraph
    //--------------------------------------------------------------------------
    OK (LAGraph_Init (msg)) ;
    //The following code taken from MIS tester
    // load the matrix as A
    const char *aname = "full.mtx";
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
    GrB_Index n = 0;
    OK (LAGraph_Cached_OutDegree (G, msg)) ;
    OK (GrB_Matrix_nrows(&n, G->A));
    //------------------------------------------------------------------
    // test the algorithm
    //------------------------------------------------------------------
    double pQ = 0;
    TEST_CHECK(
        LAGraph_SwapEdges( &G_new, &pQ, G, 100.0, msg) 
        == LAGRAPH_INSUFFICIENT_SWAPS) ;
    TEST_CHECK(pQ == 0.0);
    printf ("Test ends. \n") ;
    printf ("%s\n", msg) ;
    //------------------------------------------------------------------
    // check results (No swaps should have occured)
    //------------------------------------------------------------------

    bool ok = false;
    OK (LAGraph_Matrix_IsEqual (&ok, G_new->A, G->A, msg)) ;
    TEST_CHECK (ok) ;
    OK (LAGraph_Delete (&G_new, msg)) ;
    OK (LAGraph_Delete (&G, msg)) ;

    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------
    LAGraph_Finalize (msg) ;
    #endif
}
void test_SwapEdges_brutal (void)
{
    #if LG_SUITESPARSE_GRAPHBLAS_V10
    //--------------------------------------------------------------------------
    // start LAGraph
    //--------------------------------------------------------------------------
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {
        //The following code taken from MIS tester
        // load the matrix as A
        const char *aname = testsb [k];
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

        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;

        // all matricies I test on here are undirected with no self edges. 
        // If this changes, change to #if 1.
        #if 0
        // check if the pattern is symmetric
        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            // make the adjacency matrix symmetric
            OK (LAGraph_Cached_AT (G, msg)) ;
            OK (GrB_eWiseAdd (G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL)) ;
            G->is_symmetric_structure = LAGraph_TRUE ;
        }
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;

        // check for self-edges
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }
        #else
        TEST_CHECK (G->is_symmetric_structure) ;
        TEST_CHECK (G->nself_edges == 0) ;
        G->kind = LAGraph_ADJACENCY_UNDIRECTED ;
        #endif

        // compute the row degree
        OK (LAGraph_Cached_OutDegree (G, msg)) ;
        LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;
        //------------------------------------------------------------------
        // test the algorithm
        //------------------------------------------------------------------
        double pQ = 0.0;
        LG_BRUTAL_BURBLE (LAGraph_SwapEdges( &G_new, &pQ, G, 1.0, msg)) ;
        printf ("Test ends. Swaps per Edge: %g \n", pQ) ;
        printf ("%s\n", msg) ;

        //------------------------------------------------------------------
        // check results
        //------------------------------------------------------------------
        //Make sure we got a symetric back out:
        LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G_new, msg)) ;
                
        OK (LAGraph_Cached_NSelfEdges (G_new, msg)) ;
        TEST_CHECK (G_new->nself_edges == 0);

        // Check nvals stay the same. 
        GrB_Index edge_count, new_edge_count;
        OK (GrB_Matrix_nvals(&edge_count, G->A)) ;
        OK (GrB_Matrix_nvals(&new_edge_count, G_new->A)) ;
        TEST_CHECK(edge_count == new_edge_count);
        //next: check degrees stay the same.
        OK (LAGraph_Cached_OutDegree (G_new, msg)) ;

        bool ok = false;
        OK (LAGraph_Vector_IsEqual (
            &ok, G->out_degree, G_new->out_degree, msg)) ;
        TEST_CHECK (ok) ;
        OK (LAGraph_Delete (&G_new, msg)) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }
    //--------------------------------------------------------------------------
    // free everything and finalize LAGraph
    //--------------------------------------------------------------------------
    OK (LG_brutal_teardown (msg)) ;
    #endif
}
//----------------------------------------------------------------------------
// the make program is created by acutest, and it runs a list of tests:
//----------------------------------------------------------------------------

TEST_LIST =
{
    {"SwapEdges", test_SwapEdges},   
    {"SwapFull", test_SwapFull},   
    #if LG_BRUTAL_TESTS
    {"SwapFull", test_SwapEdges_brutal},
    #endif
    {NULL, NULL}
} ;
