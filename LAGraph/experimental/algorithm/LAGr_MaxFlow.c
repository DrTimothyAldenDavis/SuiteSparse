//------------------------------------------------------------------------------
// LAGr_MaxFlow: max flow
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Darin Peries and Tim Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGr_MaxFlow is a GraphBLAS implementation of the push-relabel algorithm
// of Baumstark et al. [1]
//
// [1] N. Baumstark, G. E. Blelloch, and J. Shun, "Efficient Implementation of
// a Synchronous Parallel Push-Relabel Algorithm." In: Bansal, N., Finocchi, I.
// (eds) Algorithms - ESA 2015. Lecture Notes in Computer Science(), vol 9294.
// Springer, Berlin, Heidelberg.  https://doi.org/10.1007/978-3-662-48350-3 10.

// [2] D. Peries and T. Davis, "A parallel push-relabel maximum flow algorithm
// in LAGraph and GraphBLAS", IEEE HPEC'25, Sept 2025.

// TODO: return the (optional) flow matrix can be costly in terms of run time.
// The HPEC'25 results only benchmark the computation of the max flow, f.
// Future work:  we plan on revising how the flow matrix is constructed.

#include <LAGraphX.h>
#include "LG_internal.h"
#include <LAGraph.h>

//#define DBG
#if LG_SUITESPARSE_GRAPHBLAS_V10

//------------------------------------------------------------------------------
// LG_augment_maxflow: sum current excess flow into the output flow
//------------------------------------------------------------------------------

#undef  LG_FREE_ALL
#define LG_FREE_ALL ;

static GrB_Info LG_augment_maxflow
(
    double *f,                  // total maxflow from src to sink
    GrB_Vector e,               // excess vector, of type double
    GrB_Index sink,             // sink node
    GrB_Vector src_and_sink,    // mask vector, with just [src sink]
    GrB_Index *n_active,        // # of active nodes
    char *msg
)
{
    // e_sink = e (sink)
    double e_sink = 0;
    GrB_Info info = GrB_Vector_extractElement(&e_sink, e, sink); //if value at sink
    GRB_TRY (info) ;
    if (info == GrB_SUCCESS)
    {
        // e(sink) is present
        (*f) += e_sink;
    }

    // TODO: what if e is tiny?  Do we need a tol parameter,
    // and replace all "e > 0" and "r > 0" comparisons with
    // (e > tol), throughout the code?

    // e<!struct([src,sink])> = select e where (e > 0)
    GRB_TRY(GrB_select(e, src_and_sink, NULL, GrB_VALUEGT_FP64, e, 0, GrB_DESC_RSC));

    // n_active = # of entries in e
    GRB_TRY(GrB_Vector_nvals(n_active, e));
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LG_global_relabel: global relabeling, based on a BFS from the sink node
//------------------------------------------------------------------------------

#undef  LG_FREE_WORK
#define LG_FREE_WORK                        \
{                                           \
    GrB_free(&R_hat);                       \
    GrB_free(&R_hat_transpose);             \
    LAGraph_Delete(&G2, msg);               \
}

#undef  LG_FREE_ALL
#define LG_FREE_ALL LG_FREE_WORK

static GrB_Info LG_global_relabel
(
    // inputs:
    GrB_Matrix R,               // flow matrix
    GrB_Index sink,             // sink node
    GrB_Vector src_and_sink,    // mask vector, with just [src sink]
    GrB_UnaryOp GetResidual,    // unary op to compute resid=capacity-flow
    GrB_BinaryOp global_relabel_accum, // accum for the disconnected assign
    GrB_Index relabel_value, // value to relabel the disconnected nodes
    // input/output:
    GrB_Vector d,       // d(i) = height/label of node i
    // outputs:
    GrB_Vector *lvl,    // lvl(i) = distance of node i from sink, if reachable
    char *msg
)
{
    GrB_Matrix R_hat = NULL, R_hat_transpose = NULL ;
    LAGraph_Graph G2 = NULL ;
    GrB_Index n ;
    GRB_TRY(GrB_Matrix_nrows(&n, R)) ;
    GRB_TRY(GrB_Matrix_new(&R_hat_transpose, GrB_FP64, n, n));
    GRB_TRY(GrB_Matrix_new(&R_hat, GrB_FP64, n, n));
    // R_hat = GetResidual (R), computing the residual of each edge
    GRB_TRY(GrB_apply(R_hat, NULL, NULL, GetResidual, R, NULL)) ;
    // prune zeros and negative entries from R_hat
    GRB_TRY(GrB_select(R_hat, NULL, NULL, GrB_VALUEGT_FP64, R_hat, 0, NULL)) ;
    // R_hat_transpose = R_hat'
    GRB_TRY(GrB_transpose(R_hat_transpose, NULL, NULL, R_hat, NULL));
    // construct G2 and its cached transpose and outdegree
    LG_TRY(LAGraph_New(&G2, &R_hat_transpose, LAGraph_ADJACENCY_DIRECTED, msg));
    G2->AT = R_hat ;
    R_hat = NULL ;
    LG_TRY(LAGraph_Cached_OutDegree(G2, msg));
    // compute lvl using bfs on G2, starting at sink node
    LG_TRY(LAGr_BreadthFirstSearch(lvl, NULL, G2, sink, msg));
    // d<!struct([src,sink])> = max(d(i), lvl)
    GRB_TRY(GrB_assign(d, src_and_sink, global_relabel_accum, *lvl, GrB_ALL, n, GrB_DESC_SC));
    // d<!struct(lvl)> = max(d(i), relabel_value)
    GRB_TRY(GrB_assign(d, *lvl, global_relabel_accum, relabel_value, GrB_ALL, n, GrB_DESC_SC));
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------

#define LG_FREE_WORK_EXCEPT_R               \
{                                           \
    GrB_free(&CompareTuple);                \
    GrB_free(&e);                           \
    GrB_free(&d);                           \
    GrB_free(&theta);                       \
    GrB_free(&Delta);                       \
    GrB_free(&delta_vec);                   \
    GrB_free(&C);                           \
    GrB_free(&push_vector);                 \
    GrB_free(&pd);                          \
    GrB_free(&src_and_sink);                \
    GrB_free(&Jvec);                        \
    GrB_free(&Prune);                       \
    GrB_free(&UpdateFlow);                  \
    GrB_free(&Relabel);                     \
    GrB_free(&ResidualFlow);                \
    GrB_free(&Cxe_IndexMult);               \
    GrB_free(&Cxe_Mult);                    \
    GrB_free(&Cxe_Semiring);                \
    GrB_free(&ExtractJ);                    \
    GrB_free(&CreateCompareVec);            \
    GrB_free(&Rxd_Semiring);                \
    GrB_free(&Rxd_Add);                     \
    GrB_free(&Rxd_AddMonoid);               \
    GrB_free(&Rxd_IndexMult);               \
    GrB_free(&Rxd_Mult);                    \
    GrB_free(&InitForw);                    \
    GrB_free(&InitBack);                    \
    GrB_free(&ResidualForward);             \
    GrB_free(&ResidualBackward);            \
    GrB_free(&zero);                        \
    GrB_free(&empty);                       \
    GrB_free(&t);                           \
    GrB_free(&invariant);                   \
    GrB_free(&CheckInvariant);              \
    GrB_free(&check);                       \
    GrB_free(&ExtractYJ);                   \
    GrB_free(&desc);                        \
    GrB_free(&MakeFlow);                    \
    GrB_free(&GetResidual);                 \
    GrB_free(&lvl) ;                        \
}

#undef  LG_FREE_WORK
#define LG_FREE_WORK                        \
{                                           \
    LG_FREE_WORK_EXCEPT_R                   \
    GrB_free(&Cxe_Add);                     \
    GrB_free(&Cxe_AddMonoid);               \
    GrB_free(&ResultTuple);                 \
    GrB_free(&FlowEdge);                    \
    GrB_free(&ExtractMatrixFlow);           \
    GrB_free(&R);                           \
}

#undef  LG_FREE_ALL
#define LG_FREE_ALL                         \
{                                           \
    LG_FREE_WORK ;                          \
    GrB_free(flow_mtx) ;                    \
}

// helper macro for creating types and operators
#define JIT_STR(f, var) char* var = #f; f

//casting for unary ops
#define F_UNARY(f) ((void (*)(void *, const void *))f)

// casting for index binary ops
#define F_INDEX_BINARY(f) ((void (*)(void*, const void*, GrB_Index, GrB_Index, const void *, GrB_Index, GrB_Index, const void *)) f)

// casting for binary op
#define F_BINARY(f) ((void (*)(void *, const void *, const void *)) f)

//------------------------------------------------------------------------------
// custom types
//------------------------------------------------------------------------------

// type of the R matrix: LG_MF_flowEdge (FlowEdge)
JIT_STR(typedef struct{
  double capacity;      /* original edge weight A(i,j), always positive */
  double flow;          /* current flow along this edge (i,j); can be negative */
  } LG_MF_flowEdge;, FLOWEDGE_STR)

// type of the push_vector vector for push_vector = R*d: LG_MF_resultTuple64/32 (ResultTuple)
JIT_STR(typedef struct{
  double residual;      /* residual = capacity - flow for the edge (i,j) */
  int64_t d;            /* d(j) of the target node j */
  int64_t j;            /* node id of the target node j */
  } LG_MF_resultTuple64;, RESULTTUPLE_STR64)
JIT_STR(typedef struct{
  double residual;      /* residual = capacity - flow for the edge (i,j) */
  int32_t d;            /* d(j) of the target node j */
  int32_t j;            /* node id of the target node j */
  } LG_MF_resultTuple32;, RESULTTUPLE_STR32)

// type of the C matrix and pd vector: LG_MF_compareTuple64/32 (CompareTuple)
JIT_STR(typedef struct{
  double residual;      /* residual = capacity - flow for the edge (i,j) */
  int64_t di;           /* d(i) for node i */
  int64_t dj;           /* d(j) for node j */
  int64_t j;            /* node id for node j */
  } LG_MF_compareTuple64;, COMPARETUPLE_STR64)
JIT_STR(typedef struct{
  double residual;      /* residual = capacity - flow for the edge (i,j) */
  int32_t di;           /* d(i) for node i */
  int32_t dj;           /* d(j) for node j */
  int32_t j;            /* node id for node j */
  int32_t unused;       /* to pad the struct to 24 bytes */
  } LG_MF_compareTuple32;, COMPARETUPLE_STR32) // 24 bytes: padded

//------------------------------------------------------------------------------
// unary ops to create R from input adjacency matrix G->A and G->AT
//------------------------------------------------------------------------------

// unary op for R = ResidualForward (A)
JIT_STR(void LG_MF_ResidualForward(LG_MF_flowEdge *z, const double *y) {
  z->capacity = (*y);
  z->flow = 0;
  }, CRF_STR)

// unary op for R<!struct(A)> = ResidualBackward (AT)
JIT_STR(void LG_MF_ResidualBackward(LG_MF_flowEdge *z, const double *y) {
  z->capacity = 0;
  z->flow = 0;
  }, CRB_STR)

//------------------------------------------------------------------------------
// R*d semiring
//------------------------------------------------------------------------------

// multiplicative operator, z = R(i,j) * d(j), 64-bit case
JIT_STR(void LG_MF_Rxd_Mult64(LG_MF_resultTuple64 *z,
    const LG_MF_flowEdge *x, GrB_Index i, GrB_Index j,
    const int64_t *y, GrB_Index iy, GrB_Index jy,
    const bool* theta) {
  double r = x->capacity - x->flow;
  if(r > 0){
    z->residual = r;
    z->d = (*y);
    z->j = j;
  }
  else{
    z->residual = 0;
    z->d = INT64_MAX;
    z->j = -1;
  }
}, RXDMULT_STR64)

// multiplicative operator, z = R(i,j) * d(j), 32-bit case
JIT_STR(void LG_MF_Rxd_Mult32(LG_MF_resultTuple32 *z,
    const LG_MF_flowEdge *x, GrB_Index i, GrB_Index j,
    const int32_t *y, GrB_Index iy, GrB_Index jy,
    const bool* theta) {
  double r = x->capacity - x->flow;
  if(r > 0){
    z->residual = r;
    z->d = (*y);
    z->j = j;
  }
  else{
    z->residual = 0;
    z->d = INT32_MAX;
    z->j = -1;
  }
}, RXDMULT_STR32)

// additive monoid: z = the best tuple, x or y, 64-bit case
JIT_STR(void LG_MF_Rxd_Add64(LG_MF_resultTuple64 * z,
    const LG_MF_resultTuple64 * x,
    const LG_MF_resultTuple64 * y) {
  if(x->d < y->d){
    (*z) = (*x) ;
  }
  else if(x->d > y->d){
    (*z) = (*y) ;
  }
  else{
    if(x->residual > y->residual){
      (*z) = (*x) ;
    }
    else if(x->residual < y->residual){
      (*z) = (*y) ;
    }
    else{
      if(x->j > y->j){
        (*z) = (*x);
      }
      else{
        (*z) = (*y) ;
      }
    }
  }
  }, RXDADD_STR64)

// additive monoid: z = the best tuple, x or y, 32-bit case
JIT_STR(void LG_MF_Rxd_Add32(LG_MF_resultTuple32 * z,
    const LG_MF_resultTuple32 * x, const LG_MF_resultTuple32 * y) {
  if(x->d < y->d){
    (*z) = (*x) ;
  }
  else if(x->d > y->d){
    (*z) = (*y) ;
  }
  else{
    if(x->residual > y->residual){
      (*z) = (*x) ;
    }
    else if(x->residual < y->residual){
      (*z) = (*y) ;
    }
    else{
      if(x->j > y->j){
        (*z) = (*x);
      }
      else{
        (*z) = (*y) ;
      }
    }
  }
  }, RXDADD_STR32)

//------------------------------------------------------------------------------
// unary ops for delta_vec = ResidualFlow (push_vector)
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_ResidualFlow64(double *z, const LG_MF_resultTuple64 *x)
    { (*z) = x->residual; }, RESIDUALFLOW_STR64)

JIT_STR(void LG_MF_ResidualFlow32(double *z, const LG_MF_resultTuple32 *x)
    { (*z) = x->residual; }, RESIDUALFLOW_STR32)

//------------------------------------------------------------------------------
// binary op for R<Delta> = UpdateFlow (R, Delta) using eWiseMult
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_UpdateFlow(LG_MF_flowEdge *z,
    const LG_MF_flowEdge *x, const double *y) {
  z->capacity = x->capacity;
  z->flow = x->flow + (*y);
  }, UPDATEFLOW_STR)

//------------------------------------------------------------------------------
// binary op for d<struct(push_vector)> = Relabel (d, push_vector)
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_Relabel64(int64_t *z,
    const int64_t *x, const LG_MF_resultTuple64 *y) {
  if((*x) < y->d+1){
    (*z) = y->d + 1;
  }
  else {
    (*z) = (*x);
  }
  }, RELABEL_STR64)

JIT_STR(void LG_MF_Relabel32(int32_t *z,
    const int32_t *x, const LG_MF_resultTuple32 *y) {
  if((*x) < y->d+1){
    (*z) = y->d + 1;
  }
  else {
    (*z) = (*x);
  }
  }, RELABEL_STR32)

//------------------------------------------------------------------------------
// unary op for Jvec = ExtractJ (pd), where Jvec(i) = pd(i)->j
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_ExtractJ64(int64_t *z, const LG_MF_compareTuple64 *x) { (*z) = x->j; }, EXTRACTJ_STR64)

JIT_STR(void LG_MF_ExtractJ32(int32_t *z, const LG_MF_compareTuple32 *x) { (*z) = x->j; }, EXTRACTJ_STR32)

//------------------------------------------------------------------------------
// unary op for Jvec = ExtractYJ(push_vector), where Jvec(i) = push_vector(i)->j
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_ExtractYJ64(int64_t *z, const LG_MF_resultTuple64 *x) { (*z) = x->j; }, EXTRACTYJ_STR64)

JIT_STR(void LG_MF_ExtractYJ32(int32_t *z, const LG_MF_resultTuple32 *x) { (*z) = x->j; }, EXTRACTYJ_STR32)

//------------------------------------------------------------------------------
// binary op for R(src,:) = InitForw (R (src,:), t')
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_InitForw(LG_MF_flowEdge * z,
    const LG_MF_flowEdge * x, const LG_MF_flowEdge * y){
  z->capacity = x->capacity;
  z->flow = y->flow + x->flow;
  }, INITFORW_STR)

//------------------------------------------------------------------------------
// binary op for R(:,src) = InitBack (R (:,src), t)
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_InitBack(LG_MF_flowEdge * z,
    const LG_MF_flowEdge * x, const LG_MF_flowEdge * y){
  z->capacity = x->capacity;
  z->flow = x->flow - y->flow;
  }, INITBACK_STR)

//------------------------------------------------------------------------------
// push_vector = C*e semiring
//------------------------------------------------------------------------------

// multiplicative operator, z = C(i,j)*e(j), 64-bit case
JIT_STR(void LG_MF_Cxe_Mult64(LG_MF_resultTuple64 * z,
    const LG_MF_compareTuple64 * x, GrB_Index i, GrB_Index j,
    const double * y, GrB_Index iy, GrB_Index jy,
    const bool* theta){
  bool j_active = ((*y) > 0) ;
  if ((x->di <  x->dj-1) /* case a */
  ||  (x->di == x->dj-1 && !j_active) /* case b */
  ||  (x->di == x->dj   && (!j_active || (j_active && (i < j)))) /* case c */
  ||  (x->di == x->dj+1))   /* case d */
  {
      z->residual = x->residual;
      z->d = x->dj;
      z->j = x->j;
  }
  else
  {
      z->residual = 0;
      z->d = INT64_MAX;
      z->j = -1;
  }
}, MXEMULT_STR64)

// multiplicative operator, z = C(i,j)*e(j), 32-bit case
JIT_STR(void LG_MF_Cxe_Mult32(LG_MF_resultTuple32 * z,
    const LG_MF_compareTuple32 * x, GrB_Index i, GrB_Index j,
    const double * y, GrB_Index iy, GrB_Index jy,
    const bool* theta){
  bool j_active = ((*y) > 0) ;
  if ((x->di <  x->dj-1) /* case a */
  ||  (x->di == x->dj-1 && !j_active) /* case b */
  ||  (x->di == x->dj && (!j_active || (j_active && (i < j)))) /* case c */
  ||  (x->di == x->dj+1))   /* case d */
  {
      z->residual = x->residual;
      z->d = x->dj;
      z->j = x->j;
  }
  else
  {
      z->residual = 0;
      z->d = INT32_MAX;
      z->j = -1;
  }
}, MXEMULT_STR32)

// Note: the additive monoid is not actually used in the call to GrB_mxv below,
// because any given node only pushes to one neighbor at a time.  As a result,
// no reduction is needed in GrB_mxv.  The semiring still needs a monoid,
// however.
JIT_STR(void LG_MF_Cxe_Add64(LG_MF_resultTuple64 * z,
    const LG_MF_resultTuple64 * x, const LG_MF_resultTuple64 * y){
    (*z) = (*y) ;
  }, MXEADD_STR64)

JIT_STR(void LG_MF_Cxe_Add32(LG_MF_resultTuple32 * z,
    const LG_MF_resultTuple32 * x, const LG_MF_resultTuple32 * y){
    (*z) = (*y) ;
  }, MXEADD_STR32)

//------------------------------------------------------------------------------
// binary op for pd = CreateCompareVec (push_vector,d) using eWiseMult
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_CreateCompareVec64(LG_MF_compareTuple64 *comp,
    const LG_MF_resultTuple64 *res, const int64_t *height) {
  comp->di = (*height);
  comp->residual = res->residual;
  comp->dj = res->d;
  comp->j = res->j;
  }, CREATECOMPAREVEC_STR64)

JIT_STR(void LG_MF_CreateCompareVec32(LG_MF_compareTuple32 *comp,
    const LG_MF_resultTuple32 *res, const int32_t *height) {
  comp->di = (*height);
  comp->residual = res->residual;
  comp->dj = res->d;
  comp->j = res->j;
  comp->unused = 0 ;
  }, CREATECOMPAREVEC_STR32)

//------------------------------------------------------------------------------
// index unary op to remove empty tuples from push_vector (for which y->j is -1)
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_Prune64(bool * z, const LG_MF_resultTuple64 * x,
  GrB_Index ix, GrB_Index jx, const bool * theta){
  *z = (x->j != -1) ;
  }, PRUNE_STR64)

JIT_STR(void LG_MF_Prune32(bool * z, const LG_MF_resultTuple32 * x,
  GrB_Index ix, GrB_Index jx, const bool * theta){
  *z = (x->j != -1) ;
  }, PRUNE_STR32)

//------------------------------------------------------------------------------
// unary op for t = MakeFlow (e), where t(i) = (0, e(i))
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_MakeFlow(LG_MF_flowEdge * flow_edge, const double * flow){
  flow_edge->capacity = 0;
  flow_edge->flow = (*flow);
  }, MAKEFLOW_STR)

//------------------------------------------------------------------------------
// binary op CheckInvariant to check invariants (debugging only)
//------------------------------------------------------------------------------

#ifdef DBG
JIT_STR(void LG_MF_CheckInvariant64(bool *z, const int64_t *height,
    const LG_MF_resultTuple64 *result) {
  (*z) = ((*height) == result->d+1);
  }, CHECKINVARIANT_STR64)

JIT_STR(void LG_MF_CheckInvariant32(bool *z, const int32_t *height,
    const LG_MF_resultTuple32 *result) {
  (*z) = ((*height) == result->d+1);
  }, CHECKINVARIANT_STR32)
#endif

//------------------------------------------------------------------------------
// binary op for R_hat = GetResidual (R), computing the residual of each edge
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_GetResidual(double * res, const LG_MF_flowEdge * flow_edge){
    (*res) = flow_edge->capacity - flow_edge->flow;
}, GETRESIDUAL_STR)

//------------------------------------------------------------------------------
// unary op for flow_mtx = ExtractMatrixFlow (R)
//------------------------------------------------------------------------------

JIT_STR(void LG_MF_ExtractMatrixFlow(double* flow, const LG_MF_flowEdge* edge){
    *flow = edge->flow;
}, EMFLOW_STR)

#endif

#ifdef DBG
void print_compareVec(const GrB_Vector vec) {
  GxB_Iterator iter;
  GxB_Iterator_new(&iter);
  GrB_Info info = GxB_Vector_Iterator_attach(iter, vec, NULL);
  if(info < 0){
    printf("error with matrix passed in");
  }
  info = GxB_Vector_Iterator_seek(iter, 0);
  while(info != GxB_EXHAUSTED){
    GrB_Index i;
    i = GxB_Vector_Iterator_getIndex(iter);
    LG_MF_compareTuple32 e;
    GxB_Iterator_get_UDT(iter, &e);
    printf("(%ld, 0)         (di: %d, dj: %d, J: %d, residual: %lf) \n", i, e.di, e.dj, e.j, e.residual);
    info = GxB_Vector_Iterator_next(iter);
  }
  GrB_free(&iter);
}
#endif


//------------------------------------------------------------------------------
// LAGraph_MaxFlow
//------------------------------------------------------------------------------

int LAGr_MaxFlow
(
    // output:
    double *f,              // max flow from src node to sink node
    GrB_Matrix *flow_mtx,   // optional output flow matrix
    // input:
    LAGraph_Graph G,        // graph to compute maxflow on
    GrB_Index src,          // source node
    GrB_Index sink,         // sink node
    char *msg
)
{

#if LG_SUITESPARSE_GRAPHBLAS_V10

  //----------------------------------------------------------------------------
  // declare variables
  //----------------------------------------------------------------------------

  // types
  GrB_Type FlowEdge = NULL ;
  GrB_Type ResultTuple = NULL ;
  GrB_Type CompareTuple = NULL ;

  GrB_Vector lvl = NULL ;
  GrB_UnaryOp GetResidual = NULL ;

  // to create R
  GrB_UnaryOp ResidualForward = NULL, ResidualBackward = NULL ;
  GrB_Matrix R = NULL ;

  // to initialize R with initial saturated flows
  GrB_Vector e = NULL, t = NULL ;
  GrB_UnaryOp MakeFlow = NULL ;
  GrB_BinaryOp InitForw = NULL, InitBack = NULL ;

  // height/label vector
  GrB_Vector d = NULL ;

  // src and sink mask vector and n_active
  GrB_Vector src_and_sink = NULL ;
  GrB_Index n_active = INT64_MAX ;

  // semiring and vectors for push_vector<struct(e)> = R*d
  GrB_Vector push_vector = NULL ;
  GrB_IndexUnaryOp Prune = NULL ;
  GxB_IndexBinaryOp Rxd_IndexMult = NULL ;
  GrB_BinaryOp Rxd_Add = NULL, Rxd_Mult = NULL ;
  GrB_Monoid Rxd_AddMonoid = NULL ;
  GrB_Semiring Rxd_Semiring = NULL ;
  GrB_Scalar theta = NULL ;

  // binary op and pd
  GrB_Vector pd = NULL ;
  GrB_BinaryOp CreateCompareVec = NULL ;

  // utility vectors, Matrix, and ops for mapping
  GrB_Matrix C = NULL ;         // matrix of candidate pushes
  GrB_Vector Jvec = NULL ;
  GrB_UnaryOp ExtractJ = NULL, ExtractYJ = NULL ;

  // C*e semiring
  GrB_Semiring Cxe_Semiring = NULL ;
  GrB_Monoid Cxe_AddMonoid = NULL ;
  GrB_BinaryOp Cxe_Add = NULL, Cxe_Mult = NULL ;
  GxB_IndexBinaryOp Cxe_IndexMult = NULL ;

  // to extract the residual flow
  GrB_UnaryOp ResidualFlow = NULL ;

  // to extract the final flows for the flow matrix
  GrB_UnaryOp ExtractMatrixFlow = NULL ;

  // Delta structures
  GrB_Vector delta_vec = NULL ;
  GrB_Matrix Delta = NULL ;

  // update height
  GrB_BinaryOp Relabel = NULL ;

  // update R structure
  GrB_BinaryOp UpdateFlow = NULL ;

  // scalars
  GrB_Scalar zero = NULL ;
  GrB_Scalar empty = NULL ;

  // invariant (for debugging only)
  GrB_Vector invariant = NULL ;
  GrB_BinaryOp CheckInvariant = NULL ;
  GrB_Scalar check = NULL ;
  bool check_raw;

  // descriptor for matrix building
  GrB_Descriptor desc = NULL ;

  //----------------------------------------------------------------------------
  // check inputs
  //----------------------------------------------------------------------------

  if (flow_mtx != NULL)
  {
    (*flow_mtx) = NULL ;
  }
  LG_TRY(LAGraph_CheckGraph(G, msg));
  LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;
  (*f) = 0;
  GrB_Index nrows, n;
  GRB_TRY(GrB_Matrix_ncols(&n, G->A));
  GRB_TRY(GrB_Matrix_nrows(&nrows, G->A));
  LG_ASSERT_MSG(nrows == n, GrB_INVALID_VALUE, "Matrix must be square");
  LG_ASSERT_MSG(src < n && src >= 0 && sink < n && sink >= 0,
        GrB_INVALID_VALUE, "src and sink must be a value between [0, n)");
  LG_ASSERT_MSG(G->emin > 0, GrB_INVALID_VALUE,
        "the edge weights (capacities) must be greater than 0");

  //get adjacency matrix and its transpose
  GrB_Matrix A = G->A;
  GrB_Matrix AT = NULL ;
  if (G->kind == LAGraph_ADJACENCY_UNDIRECTED)
  {
    // G is undirected, so A and AT are the same
    AT = G->A ;
  }
  else
  {
    // G is directed; get G->AT, which must be present
    AT = G->AT ;
    LG_ASSERT_MSG (AT != NULL, LAGRAPH_NOT_CACHED, "G->AT is required") ;
  }

  //----------------------------------------------------------------------------
  // create types, operators, matrices, and vectors
  //----------------------------------------------------------------------------

  // create types for computation
  GRB_TRY(GxB_Type_new(&FlowEdge, sizeof(LG_MF_flowEdge), "LG_MF_flowEdge", FLOWEDGE_STR));

  GRB_TRY(GxB_UnaryOp_new(&GetResidual, F_UNARY(LG_MF_GetResidual), GrB_FP64, FlowEdge,
        "LG_MF_GetResidual", GETRESIDUAL_STR));

  #ifdef DBG
  GRB_TRY(GrB_Scalar_new(&check, GrB_BOOL));
  GRB_TRY(GrB_Scalar_setElement(check, false));
  GRB_TRY(GrB_Vector_new(&invariant, GrB_BOOL, n));
  #endif

  // ops create R from A
  GRB_TRY(GxB_UnaryOp_new(&ResidualForward,
        F_UNARY(LG_MF_ResidualForward), FlowEdge , GrB_FP64,
        "LG_MF_ResidualForward", CRF_STR));
  GRB_TRY(GxB_UnaryOp_new(&ResidualBackward,
        F_UNARY(LG_MF_ResidualBackward), FlowEdge , GrB_FP64,
        "LG_MF_ResidualBackward", CRB_STR));

  // ops to initialize R with initial saturated flows from the source node
  GRB_TRY(GxB_BinaryOp_new(&InitForw,
        F_BINARY(LG_MF_InitForw), FlowEdge, FlowEdge, FlowEdge,
        "LG_MF_InitForw", INITFORW_STR));
  GRB_TRY(GxB_BinaryOp_new(&InitBack,
        F_BINARY(LG_MF_InitBack), FlowEdge, FlowEdge, FlowEdge,
        "LG_MF_InitBack", INITBACK_STR));
  GRB_TRY(GxB_UnaryOp_new(&MakeFlow, F_UNARY(LG_MF_MakeFlow), FlowEdge, GrB_FP64,
        "LG_MF_MakeFlow", MAKEFLOW_STR));

  // construct [src,sink] mask
  GRB_TRY(GrB_Vector_new(&src_and_sink, GrB_BOOL, n));
  GRB_TRY (GrB_Vector_setElement (src_and_sink, true, sink)) ;
  GRB_TRY (GrB_Vector_setElement (src_and_sink, true, src)) ;

  // create delta vector and Delta matrix
  GRB_TRY(GrB_Matrix_new(&Delta, GrB_FP64, n, n));
  GRB_TRY(GrB_Vector_new(&delta_vec, GrB_FP64, n));

  // operator to update R structure
  GRB_TRY(GxB_BinaryOp_new(&UpdateFlow,
        F_BINARY(LG_MF_UpdateFlow), FlowEdge, FlowEdge, GrB_FP64,
        "LG_MF_UpdateFlow", UPDATEFLOW_STR));

  // create scalars
  GRB_TRY(GrB_Scalar_new(&zero, GrB_FP64));
  GRB_TRY(GrB_Scalar_setElement(zero, 0));
  GRB_TRY(GrB_Scalar_new (&empty, GrB_FP64)) ;
  GRB_TRY(GrB_Scalar_new(&theta, GrB_BOOL));        // unused placeholder
  GRB_TRY(GrB_Scalar_setElement_BOOL(theta, false));

  //----------------------------------------------------------------------------
  // determine the integer type to use for the problem
  //----------------------------------------------------------------------------

  GrB_Type Integer_Type = NULL ;

  //accum operator for the global relabel
  GrB_BinaryOp global_relabel_accum = NULL ;

  #ifdef COVERAGE
  // Just for test coverage, use 64-bit ints for n > 100.  Do not use this
  // rule in production!
  #define NBIG 100
  #else
  // For production use: 64-bit integers if n > 2^31
  #define NBIG INT32_MAX
  #endif
  if (n > NBIG){

    //--------------------------------------------------------------------------
    // use 64-bit integers
    //--------------------------------------------------------------------------

    Integer_Type = GrB_INT64 ;

    // use the 64 bit max operator 
    global_relabel_accum = GrB_MAX_INT64 ;

    // create types for computation
    GRB_TRY(GxB_Type_new(&ResultTuple, sizeof(LG_MF_resultTuple64),
        "LG_MF_resultTuple64", RESULTTUPLE_STR64));
    GRB_TRY(GxB_Type_new(&CompareTuple, sizeof(LG_MF_compareTuple64),
        "LG_MF_compareTuple64", COMPARETUPLE_STR64));

    // invariant check
    #ifdef DBG
    GRB_TRY(GxB_BinaryOp_new(&CheckInvariant,
        F_BINARY(LG_MF_CheckInvariant64), GrB_BOOL, GrB_INT64, ResultTuple,
        "LG_MF_CheckInvariant64", CHECKINVARIANT_STR64));
    #endif

    GRB_TRY(GxB_UnaryOp_new(&ResidualFlow,
        F_UNARY(LG_MF_ResidualFlow64), GrB_FP64, ResultTuple,
        "LG_MF_ResidualFlow64", RESIDUALFLOW_STR64));

    // create ops for R*d semiring

    GRB_TRY(GxB_IndexBinaryOp_new(&Rxd_IndexMult,
        F_INDEX_BINARY(LG_MF_Rxd_Mult64), ResultTuple, FlowEdge, GrB_INT64, GrB_BOOL,
        "LG_MF_Rxd_Mult64", RXDMULT_STR64));
    GRB_TRY(GxB_BinaryOp_new_IndexOp(&Rxd_Mult, Rxd_IndexMult, theta));
    GRB_TRY(GxB_BinaryOp_new(&Rxd_Add,
        F_BINARY(LG_MF_Rxd_Add64), ResultTuple, ResultTuple, ResultTuple,
        "LG_MF_Rxd_Add64", RXDADD_STR64));
    LG_MF_resultTuple64 id = {.d = INT64_MAX, .j = -1, .residual = 0};
    GRB_TRY(GrB_Monoid_new_UDT(&Rxd_AddMonoid, Rxd_Add, &id));

    // create binary op for pd
    GRB_TRY(GxB_BinaryOp_new(&CreateCompareVec,
        F_BINARY(LG_MF_CreateCompareVec64), CompareTuple, ResultTuple, GrB_INT64,
        "LG_MF_CreateCompareVec64", CREATECOMPAREVEC_STR64));

    // create op to prune empty tuples
    GRB_TRY(GxB_IndexUnaryOp_new(&Prune,
        (GxB_index_unary_function) LG_MF_Prune64, GrB_BOOL, ResultTuple, GrB_BOOL,
        "LG_MF_Prune64", PRUNE_STR64));

    // create ops for mapping
    GRB_TRY(GxB_UnaryOp_new(&ExtractJ,
        F_UNARY(LG_MF_ExtractJ64), GrB_INT64, CompareTuple,
        "LG_MF_ExtractJ64", EXTRACTJ_STR64));
    GRB_TRY(GxB_UnaryOp_new(&ExtractYJ,
        F_UNARY(LG_MF_ExtractYJ64), GrB_INT64, ResultTuple,
        "LG_MF_ExtractYJ64", EXTRACTYJ_STR64));

    // create ops for C*e semiring
    GRB_TRY(GxB_IndexBinaryOp_new(&Cxe_IndexMult,
        F_INDEX_BINARY(LG_MF_Cxe_Mult64), ResultTuple, CompareTuple, GrB_FP64, GrB_BOOL,
        "LG_MF_Cxe_Mult64", MXEMULT_STR64));
    GRB_TRY(GxB_BinaryOp_new_IndexOp(&Cxe_Mult, Cxe_IndexMult, theta));
    GRB_TRY(GxB_BinaryOp_new(&Cxe_Add,
        F_BINARY(LG_MF_Cxe_Add64), ResultTuple, ResultTuple, ResultTuple,
        "LG_MF_Cxe_Add64", MXEADD_STR64));
    GRB_TRY(GrB_Monoid_new_UDT(&Cxe_AddMonoid, Cxe_Add, &id));

    // update height binary op
    GRB_TRY(GxB_BinaryOp_new(&Relabel,
        F_BINARY(LG_MF_Relabel64), GrB_INT64, GrB_INT64, ResultTuple,
        "LG_MF_Relabel64", RELABEL_STR64));

  }else{

    //--------------------------------------------------------------------------
    // use 32-bit integers
    //--------------------------------------------------------------------------

    Integer_Type = GrB_INT32 ;

    // use 32 bit max op
    global_relabel_accum = GrB_MAX_INT32 ;

    // create types for computation
    GRB_TRY(GxB_Type_new(&ResultTuple, sizeof(LG_MF_resultTuple32),
        "LG_MF_resultTuple32", RESULTTUPLE_STR32));
    GRB_TRY(GxB_Type_new(&CompareTuple, sizeof(LG_MF_compareTuple32),
        "LG_MF_compareTuple32", COMPARETUPLE_STR32));

    // invariant check
    #ifdef DBG
    GRB_TRY(GxB_BinaryOp_new(&CheckInvariant,
        F_BINARY(LG_MF_CheckInvariant32), GrB_BOOL, GrB_INT32, ResultTuple,
        "LG_MF_CheckInvariant32", CHECKINVARIANT_STR32));
    #endif

    GRB_TRY(GxB_UnaryOp_new(&ResidualFlow,
        F_UNARY(LG_MF_ResidualFlow32), GrB_FP64, ResultTuple,
        "LG_MF_ResidualFlow32", RESIDUALFLOW_STR32));

    // create ops for R*d semiring
    GRB_TRY(GxB_IndexBinaryOp_new(&Rxd_IndexMult,
        F_INDEX_BINARY(LG_MF_Rxd_Mult32), ResultTuple, FlowEdge, GrB_INT32, GrB_BOOL,
        "LG_MF_Rxd_Mult32", RXDMULT_STR32));
    GRB_TRY(GxB_BinaryOp_new_IndexOp(&Rxd_Mult, Rxd_IndexMult, theta));
    GRB_TRY(GxB_BinaryOp_new(&Rxd_Add,
        F_BINARY(LG_MF_Rxd_Add32), ResultTuple, ResultTuple, ResultTuple,
        "LG_MF_Rxd_Add32", RXDADD_STR32));
    LG_MF_resultTuple32 id = {.d = INT32_MAX, .j = -1, .residual = 0};
    GRB_TRY(GrB_Monoid_new_UDT(&Rxd_AddMonoid, Rxd_Add, &id));

    // create binary op for pd
    GRB_TRY(GxB_BinaryOp_new(&CreateCompareVec,
        F_BINARY(LG_MF_CreateCompareVec32), CompareTuple, ResultTuple, GrB_INT32,
        "LG_MF_CreateCompareVec32", CREATECOMPAREVEC_STR32));

    // create op to prune empty tuples
    GRB_TRY(GxB_IndexUnaryOp_new(&Prune,
        (GxB_index_unary_function) LG_MF_Prune32, GrB_BOOL, ResultTuple, GrB_BOOL,
        "LG_MF_Prune32", PRUNE_STR32));

    // create ops for mapping
    GRB_TRY(GxB_UnaryOp_new(&ExtractJ,
        F_UNARY(LG_MF_ExtractJ32), GrB_INT32, CompareTuple,
        "LG_MF_ExtractJ32", EXTRACTJ_STR32));
    GRB_TRY(GxB_UnaryOp_new(&ExtractYJ,
        F_UNARY(LG_MF_ExtractYJ32), GrB_INT32, ResultTuple,
        "LG_MF_ExtractYJ32", EXTRACTYJ_STR32));

    // create ops for C*e semiring
    GRB_TRY(GxB_IndexBinaryOp_new(&Cxe_IndexMult,
        F_INDEX_BINARY(LG_MF_Cxe_Mult32), ResultTuple, CompareTuple, GrB_FP64, GrB_BOOL,
        "LG_MF_Cxe_Mult32", MXEMULT_STR32));
    GRB_TRY(GxB_BinaryOp_new_IndexOp(&Cxe_Mult, Cxe_IndexMult, theta));
    GRB_TRY(GxB_BinaryOp_new(&Cxe_Add,
        F_BINARY(LG_MF_Cxe_Add32), ResultTuple, ResultTuple, ResultTuple,
        "LG_MF_Cxe_Add32", MXEADD_STR32));
    GRB_TRY(GrB_Monoid_new_UDT(&Cxe_AddMonoid, Cxe_Add, &id));

    // update height binary op
    GRB_TRY(GxB_BinaryOp_new(&Relabel,
        F_BINARY(LG_MF_Relabel32), GrB_INT32, GrB_INT32, ResultTuple,
        "LG_MF_Relabel32", RELABEL_STR32));
  }

  //----------------------------------------------------------------------------
  // create remaining vectors, matrices, descriptor, and semirings
  //----------------------------------------------------------------------------

  GRB_TRY(GrB_Matrix_new(&C, CompareTuple, n,n));
  GRB_TRY(GrB_Vector_new(&Jvec, Integer_Type, n));
  GRB_TRY(GrB_Vector_new(&pd, CompareTuple, n));
  GRB_TRY(GrB_Vector_new(&push_vector, ResultTuple, n));

  GRB_TRY(GrB_Semiring_new(&Rxd_Semiring, Rxd_AddMonoid, Rxd_Mult));
  GRB_TRY(GrB_Semiring_new(&Cxe_Semiring, Cxe_AddMonoid, Cxe_Mult));

  // create descriptor for building the C and Delta matrices
  GRB_TRY(GrB_Descriptor_new(&desc));
  GRB_TRY(GrB_set(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));

  // create and init d vector
  GRB_TRY(GrB_Vector_new(&d, Integer_Type, n));
  GRB_TRY(GrB_assign(d, NULL, NULL, 0, GrB_ALL, n, NULL));
  GRB_TRY(GrB_assign(d, NULL, NULL, n, &src, 1, NULL));

  // create R, with no flow
  GRB_TRY(GrB_Matrix_new(&R, FlowEdge, n, n));
  // R = ResidualForward (A)
  GRB_TRY(GrB_apply(R, NULL, NULL, ResidualForward, A, NULL));
  // R<!struct(A)> = ResidualBackward (AT)
  GRB_TRY(GrB_apply(R, A, NULL, ResidualBackward, AT, GrB_DESC_SC));

  // initial global relabeling
  // relabel to 2*n to prevent any flow from going to the
  // disconnected nodes.
  GrB_Index relabel_value = 2*n ;
  LG_TRY (LG_global_relabel (R, sink, src_and_sink, GetResidual, global_relabel_accum, relabel_value, d, &lvl, msg)) ;

  // reset to n
  relabel_value = n ;
  
  // create excess vector e and initial flows from the src to its neighbors
  // e<struct(lvl)> = A (src,:)
  GRB_TRY(GrB_Vector_new(&e, GrB_FP64, n));
  GRB_TRY(GrB_extract(e, lvl, NULL, A, GrB_ALL, n, src, GrB_DESC_ST0));
  GrB_free(&lvl);
  // t = MakeFlow (e), where t(i) = (0, e(i))
  GRB_TRY(GrB_Vector_new(&t, FlowEdge, n));
  GRB_TRY(GrB_apply(t, NULL, NULL, MakeFlow, e, NULL));
  // R(src,:) = InitForw (R (src,:), t')
  GRB_TRY(GrB_assign(R, NULL, InitForw, t, src, GrB_ALL, n, NULL));
  // R(:,src) = InitBack (R (:,src), t)
  GRB_TRY(GrB_assign(R, NULL, InitBack, t, GrB_ALL, n, src, NULL));
  GrB_free(&t) ;

  // augment the maxflow with the initial flows from the src to its neighbors
  LG_TRY (LG_augment_maxflow (f, e, sink, src_and_sink, &n_active, msg)) ;

  //----------------------------------------------------------------------------
  // compute the max flow
  //----------------------------------------------------------------------------

  for (int64_t iter = 0 ; n_active > 0 ; iter++)
  {
    #ifdef DBG
      printf ("iter: %ld, n_active %ld\n", iter, n_active) ;
    #endif
    //--------------------------------------------------------------------------
    // Part 1: global relabeling
    //--------------------------------------------------------------------------

    if ((iter > 0)  && (iter % 12 == 0))
    {
      #ifdef DBG
        printf ("relabel at : %ld\n", iter) ;
      #endif
      LG_TRY (LG_global_relabel (R, sink, src_and_sink, GetResidual, global_relabel_accum, relabel_value, d, &lvl, msg)) ;
      if(flow_mtx == NULL){
        // delete nodes in e that cannot be reached from the sink
	//  e<!struct(lvl)> = empty scalar
	GrB_assign (e, lvl, NULL, empty, GrB_ALL, n, GrB_DESC_SC) ;
	GRB_TRY(GrB_Vector_nvals(&n_active, e));
	if(n_active == 0) break;
      }
      GrB_free(&lvl);
    }

    //--------------------------------------------------------------------------
    // Part 2: deciding where to push
    //--------------------------------------------------------------------------

    // push_vector<struct(e),replace> = R*d using the Rxd_Semiring
    GRB_TRY(GrB_mxv(push_vector, e, NULL, Rxd_Semiring, R, d, GrB_DESC_RS));

    // remove empty tuples (0,inf,-1) from push_vector
    GRB_TRY(GrB_select(push_vector, NULL, NULL, Prune, push_vector, 0, NULL));

    //--------------------------------------------------------------------------
    // Part 3: verifying the pushes
    //--------------------------------------------------------------------------

    // create C matrix (Candidate pushes) from pattern and values of pd
    // pd = CreateCompareVec (push_vector,d) using eWiseMult
    GRB_TRY(GrB_eWiseMult(pd, NULL, NULL, CreateCompareVec, push_vector, d, NULL));

    #ifdef DBG
      print_compareVec(pd);
      GxB_print(d, 3);
      GxB_print(e, 3);
    #endif
    
    // Jvec = ExtractJ (pd), where Jvec(i) = pd(i)->j
    GRB_TRY(GrB_apply(Jvec, NULL, NULL, ExtractJ, pd, NULL));
    GRB_TRY(GrB_Matrix_clear(C));
    GRB_TRY(GrB_Matrix_build(C, pd, Jvec, pd, GxB_IGNORE_DUP, desc));
    GRB_TRY(GrB_Vector_clear(pd));
    GRB_TRY(GrB_Vector_clear(Jvec));

    // make e dense for C computation
    // TODO: consider keeping e in bitmap/full format only,
    // or always full with e(i)=0 denoting a non-active node.
    GRB_TRY(GrB_assign(e, e, NULL, 0, GrB_ALL, n, GrB_DESC_SC));

    // push_vector = C*e using the Cxe_Semiring
    GRB_TRY(GrB_mxv(push_vector, NULL, NULL, Cxe_Semiring, C, e, NULL));
    GRB_TRY(GrB_Matrix_clear(C));

    // remove empty tuples (0,inf,-1) from push_vector
    GRB_TRY(GrB_select(push_vector, NULL, NULL, Prune, push_vector, -1, NULL));

    // relabel, updating the height/label vector d
    // d<struct(push_vector)> = Relabel (d, push_vector) using eWiseMult
    GRB_TRY(GrB_eWiseMult(d, push_vector, NULL, Relabel, d, push_vector, GrB_DESC_S));

    #ifdef DBG
        // assert invariant for all labels
        GRB_TRY(GrB_eWiseMult(invariant, push_vector, NULL, CheckInvariant, d, push_vector, GrB_DESC_RS));
        GRB_TRY(GrB_reduce(check, NULL, GrB_LAND_MONOID_BOOL, invariant, NULL));
        GRB_TRY(GrB_Scalar_extractElement(&check_raw, check));
        ASSERT(check_raw == true);
    #endif

    //--------------------------------------------------------------------------
    // Part 4: executing the pushes
    //--------------------------------------------------------------------------

    // extract residual flows from push_vector
    // delta_vec = ResidualFlow (push_vector), obtaining just the residual flows
    GRB_TRY(GrB_apply(delta_vec, NULL, NULL, ResidualFlow, push_vector, NULL));

    // delta_vec = min (delta_vec, e), where e is dense
    GRB_TRY(GrB_eWiseMult(delta_vec, NULL, NULL, GrB_MIN_FP64, delta_vec, e, NULL));

    // create the Delta matrix from delta_vec and push_vector
    // note that delta_vec has the same structure as push_vector
    // Jvec = ExtractYJ (push_vector), where Jvec(i) = push_vector(i)->j
    // if Jvec has no values, then there is no possible
    // candidates to push to, so the algorithm terminates
    GRB_TRY(GrB_apply(Jvec, NULL, NULL, ExtractYJ, push_vector, NULL));
    GrB_Index J_n;
    GRB_TRY(GrB_Vector_nvals(&J_n, Jvec));
    if(J_n == 0) break;
    GRB_TRY(GrB_Matrix_clear(Delta));
    GRB_TRY(GrB_Matrix_build(Delta, delta_vec, Jvec, delta_vec, GxB_IGNORE_DUP, desc));
    GRB_TRY(GrB_Vector_clear(Jvec));

    // make Delta anti-symmetric
    // Delta = (Delta - Delta')
    GRB_TRY(GxB_eWiseUnion(Delta, NULL, NULL, GrB_MINUS_FP64, Delta, zero, Delta, zero, GrB_DESC_T1));

    // update R
    // R<Delta> = UpdateFlow (R, Delta) using eWiseMult
    GRB_TRY(GrB_eWiseMult(R, Delta, NULL, UpdateFlow, R, Delta, GrB_DESC_S));

    // reduce Delta to delta_vec
    // delta_vec = sum (Delta), summing up each row of Delta
    GRB_TRY(GrB_reduce(delta_vec, NULL, NULL, GrB_PLUS_MONOID_FP64, Delta, GrB_DESC_T0));
    GRB_TRY(GrB_Matrix_clear(Delta));

    // add delta_vec to e
    // e<struct(delta_vec)> += delta_vec
    GRB_TRY(GrB_assign(e, delta_vec, GrB_PLUS_FP64, delta_vec, GrB_ALL, n, GrB_DESC_S));

    // augment maxflow for all active nodes
    LG_TRY (LG_augment_maxflow (f, e, sink, src_and_sink, &n_active, msg)) ;

  }


  //----------------------------------------------------------------------------
  // optionally construct the output flow matrix, if requested
  //----------------------------------------------------------------------------

  if (flow_mtx != NULL)
  {
    // free all workspace except R
    LG_FREE_WORK_EXCEPT_R ;
    // create the ExtractMatrixFlow op to compute the flow matrix
    GRB_TRY(GxB_UnaryOp_new(&ExtractMatrixFlow,
        F_UNARY(LG_MF_ExtractMatrixFlow), GrB_FP64, FlowEdge,
        "LG_MF_ExtractMatrixFlow", EMFLOW_STR));
    // flow_mtx = ExtractMatrixFlow (R)
    GRB_TRY(GrB_Matrix_new(flow_mtx, GrB_FP64, n, n));
    GRB_TRY(GrB_apply(*flow_mtx, NULL, NULL, ExtractMatrixFlow, R, NULL));
    // delete any zero or negative flows from the flow_mtx
    GRB_TRY(GrB_select(*flow_mtx, NULL, NULL, GrB_VALUEGT_FP64, *flow_mtx, 0, NULL));
  }

  //----------------------------------------------------------------------------
  // for test coverage only
  //----------------------------------------------------------------------------

  #ifdef COVERAGE
  // The Cxe_Add operator is not tested via the call to GrB_mxv with the
  // Cxe_Semiring above, so test it via the Cxe_AddMonoid.
  GrB_free(&push_vector);
  GRB_TRY(GrB_Vector_new(&push_vector, ResultTuple, 3));
  if (n > NBIG)
  {
    LG_MF_resultTuple64 a = {.d = 1, .j = 2, .residual = 3};
    LG_MF_resultTuple64 b = {.d = 4, .j = 5, .residual = 6};
    GRB_TRY (GrB_Vector_setElement_UDT (push_vector, (void *) &a, 0)) ;
    GRB_TRY (GrB_Vector_setElement_UDT (push_vector, (void *) &b, 0)) ;
    LG_MF_resultTuple64 c = {.d = 0, .j = 0, .residual = 0};
    GRB_TRY (GrB_Vector_reduce_UDT ((void *) &c, NULL, Cxe_AddMonoid, push_vector, NULL)) ;
    LG_ASSERT ((c.residual == 6 && c.j == 5 && c.d == 4), GrB_PANIC) ;
  }
  else
  {
    LG_MF_resultTuple32 a = {.d = 1, .j = 2, .residual = 3};
    LG_MF_resultTuple32 b = {.d = 4, .j = 5, .residual = 6};
    GRB_TRY (GrB_Vector_setElement_UDT (push_vector, (void *) &a, 0)) ;
    GRB_TRY (GrB_Vector_setElement_UDT (push_vector, (void *) &b, 0)) ;
    LG_MF_resultTuple32 c = {.d = 0, .j = 0, .residual = 0};
    GRB_TRY (GrB_Vector_reduce_UDT ((void *) &c, NULL, Cxe_AddMonoid, push_vector, NULL)) ;
    LG_ASSERT ((c.residual == 6 && c.j == 5 && c.d == 4), GrB_PANIC) ;
  }
  #endif

  //----------------------------------------------------------------------------
  // free workspace and return result
  //----------------------------------------------------------------------------

  LG_FREE_WORK ;
  return GrB_SUCCESS;
#else
  return GrB_NOT_IMPLEMENTED ;
#endif
}
