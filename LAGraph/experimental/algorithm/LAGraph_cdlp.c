//------------------------------------------------------------------------------
// LAGraph_cdlp: community detection using label propagation
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Gabor Szarnyas and Balint Hegyi, Budapest University of
// Technology and Economics (with accented characters: G\'{a}bor Sz\'{a}rnyas
// and B\'{a}lint Hegyi, using LaTeX syntax).
// https://inf.mit.bme.hu/en/members/szarnyasg .

// Modified by Pascal Costanza, Intel, Belgium

// NOTE: the calloc/free below must be thread-safe,
// so it cannot use LAGraph_Malloc/LAGraph_Free (which can use
// the Rapids Memory Manager methods when using CUDA, and those
// methods are not yet thread-safe).

//------------------------------------------------------------------------------

// ## Background
//
// This function was originally written for the LDBC Graphalytics benchmark.
//
// The community detection using label propagation (CDLP) algorithm is
// defined both for directed and undirected graphs.
//
// The definition implemented here is described in the following document:
// https://ldbc.github.io/ldbc_graphalytics_docs/graphalytics_spec.pdf
//
// The algorithm is based on the one given in the following paper:
//
// Usha Raghavan, Reka Albert, and Soundar Kumara. "Near linear time algorithm
// to detect community structures in large-scale networks". In: Physical
// Review E 76.3 (2007), p. 036106, https://arxiv.org/abs/0709.2938
//
// The key idea of the algorithm is that each vertex is assigned the label
// that is most frequent among its neighbors. To allow reproducible
// experiments, the algorithm is modified to guarantee deterministic behavior:
// it always picks the smallest label in case of a tie:
//
// min ( argmax_{l} (#neighbors with label l) )
//
// In other words, we need to compute the *minimum mode value* (minmode) for
// the labels among the neighbors.
//
// For directed graphs, a label on a neighbor that is connected through both
// an outgoing and on an incoming edge counts twice:
//
// min ( argmax_{l} (#incoming neighbors with l + #outgoing neighbors with l) )

#define LG_FREE_ALL                                                     \
{                                                                       \
    GrB_free (&S) ;                                                     \
    GrB_free (&T) ;                                                     \
    LAGraph_Free ((void **) &Sp, NULL) ;                                \
    LAGraph_Free ((void **) &Si, NULL) ;                                \
    LAGraph_Free ((void **) &Tp, NULL) ;                                \
    LAGraph_Free ((void **) &Ti, NULL) ;                                \
    LAGraph_Free ((void **) &L, NULL) ;                                 \
    LAGraph_Free ((void **) &L_next, NULL) ;                            \
    ptable_pool_free (counts_pool, max_threads) ;                       \
    counts_pool = NULL ;                                                \
    GrB_free (&CDLP) ;                                                  \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#if LAGRAPH_HAS_STDALIGN_H
    #include <stdalign.h>
    #define ALIGNAS_64 alignas(64)
#else
    // the alignas keyword/macro is not available
    #define ALIGNAS_64
#endif
#include "LG_internal.h"

// A Go-style slice / Lisp-style property list
typedef struct {
    GrB_Index* entries;
    size_t len, cap;
} plist;

void plist_free(plist *list) {
    free(list->entries);        // NOTE: cannot be LAGraph_Free
}

void plist_clear(plist *list) {
    list->len = 0;
}

void plist_append(plist* list, GrB_Index key, GrB_Index value) {
    if (list->len == list->cap) {
        size_t new_size = list->cap == 0 ? 16 : 2*list->cap;
        list->entries = (GrB_Index*)realloc(list->entries, new_size * sizeof(GrB_Index));
        list->cap = new_size;
    }
    list->entries[list->len] = key;
    list->entries[list->len+1] = value;
    list->len += 2;
}

GrB_Index plist_add(plist* list, GrB_Index entry) {
    for (size_t i = 0; i < list->len; i += 2) {
        if (list->entries[i] == entry) {
            return ++list->entries[i+1];
        }
    }
    plist_append(list, entry, 1);
    return 1;
}

typedef void (*plist_reducer) (GrB_Index* entry1, GrB_Index* count1, GrB_Index entry2, GrB_Index count2);

void plist_reduce(plist* list, GrB_Index* entry, GrB_Index* count, plist_reducer reducer) {
    for (size_t i = 0; i < list->len; i += 2) {
        reducer(entry, count, list->entries[i], list->entries[i+1]);
    }
}

void counts_reducer(GrB_Index* e1, GrB_Index* c1, GrB_Index e2, GrB_Index c2) {
    if (*c1 > c2) {
        return;
    }
    if (c2 > *c1) {
        *e1 = e2;
        *c1 = c2;
        return;
    }
    if (*e1 < e2) {
        return;
    }
    *e1 = e2;
    *c1 = c2;
}


#define bucket_bits 9llu
#define nof_buckets (1llu << bucket_bits)
#define bucket_shift (64llu - bucket_bits)

typedef struct {
    ALIGNAS_64 plist buckets[nof_buckets];
} ptable;

void ptable_free(ptable* table) {
    for (size_t i = 0; i < nof_buckets; i++) {
        plist_free(&table->buckets[i]);
    }
}

void ptable_pool_free(ptable* table, size_t n) {
    if (table == NULL) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        ptable_free(&table[i]);
    }
    free(table);        // NOTE: cannot be LAGraph_Free
}

void ptable_clear(ptable* table) {
    for (size_t i = 0; i < nof_buckets; i++) {
        plist_clear(&table->buckets[i]);
    }
}

GrB_Index fib_reduce(GrB_Index x)
{
    // 2^64 / golden ratio = 11400714819323198485
    GrB_Index fibhash = x * 11400714819323198485llu;
    // fast reduce
    return fibhash >> bucket_shift;
}

void ptable_add(ptable* table, GrB_Index entry) {
    plist_add(&table->buckets[fib_reduce(entry)], entry);
}

void ptable_reduce(ptable* table, GrB_Index* entry, GrB_Index* count, plist_reducer reducer) {
    *entry = GrB_INDEX_MAX + 1;
    *count = 0;
    for (GrB_Index i = 0; i < nof_buckets; i++) {
        plist_reduce(&table->buckets[i], entry, count, reducer);
    }
}

//****************************************************************************

int LAGraph_cdlp
        (
                GrB_Vector *CDLP_handle,    // output vector
                LAGraph_Graph G,            // input graph
                int itermax,                // max number of iterations
                char *msg
        )
{
    GrB_Info info;
    LG_CLEAR_MSG ;

    GrB_Matrix S = NULL, T = NULL ;
    GrB_Vector CDLP ;
    GrB_Index *Sp = NULL, *Si = NULL, *Tp = NULL, *Ti = NULL, *L = NULL, *L_next = NULL ;
    ptable *counts_pool = NULL ;

    #ifdef _OPENMP
    size_t max_threads = omp_get_max_threads();
    #else
    size_t max_threads = 1 ;
    #endif

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (CDLP_handle == NULL)
    {
        return GrB_NULL_POINTER;
    }

    GrB_Matrix A = G->A ;

    //--------------------------------------------------------------------------
    // ensure input is binary and has no self-edges
    //--------------------------------------------------------------------------

    GrB_Index n;
    GRB_TRY (GrB_Matrix_nrows(&n, A)) ;

    GRB_TRY (GrB_Matrix_new (&S, GrB_UINT64, n, n)) ;
    GRB_TRY (GrB_apply (S, GrB_NULL, GrB_NULL, GrB_ONEB_UINT64, A, 0, GrB_NULL)) ;

    if (G->kind == LAGraph_ADJACENCY_DIRECTED) {
        GRB_TRY (GrB_Matrix_new (&T, GrB_UINT64, n, n)) ;
        GRB_TRY (GrB_transpose (T, GrB_NULL, GrB_NULL, S, GrB_NULL)) ;
        void * Tx = NULL ;
        GrB_Index Tps, Tis, Txs ;
#if LAGRAPH_SUITESPARSE
        bool Tiso, Tjumbled ;
        GRB_TRY (GxB_Matrix_unpack_CSR (T, &Tp, &Ti, &Tx, &Tps, &Tis, &Txs, &Tiso, &Tjumbled, GrB_NULL)) ;
#else
        GRB_TRY (GrB_Matrix_exportSize (&Tps, &Tis, &Txs, GrB_CSR_FORMAT, T)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void *)&Tp, Tps, sizeof(GrB_Index), NULL)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void *)&Ti, Tis, sizeof(GrB_Index), NULL)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void *)&Tx, Txs, sizeof(GrB_UINT64), NULL)) ;
        GRB_TRY (GrB_Matrix_export (Tp, Ti, (uint64_t *)Tx, &Tps, &Tis, &Txs, GrB_CSR_FORMAT, T)) ;
#endif
        LAGRAPH_TRY (LAGraph_Free ((void *)&Tx, NULL)) ;
        GRB_TRY (GrB_free (&T)) ;
    }

    {
        void * Sx = NULL ;
        GrB_Index Sps, Sis, Sxs ;
#if LAGRAPH_SUITESPARSE
        bool Siso, Sjumbled ;
        GRB_TRY (GxB_Matrix_unpack_CSR (S, &Sp, &Si, &Sx, &Sps, &Sis, &Sxs, &Siso, &Sjumbled, GrB_NULL)) ;
#else
        GRB_TRY (GrB_Matrix_exportSize (&Sps, &Sis, &Sxs, GrB_CSR_FORMAT, S)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void *)&Sp, Sps, sizeof(GrB_Index), NULL)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void *)&Si, Sis, sizeof(GrB_Index), NULL)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void *)&Sx, Sxs, sizeof(GrB_UINT64), NULL)) ;
        GRB_TRY (GrB_Matrix_export (Sp, Si, (uint64_t *)Sx, &Sps, &Sis, &Sxs, GrB_CSR_FORMAT, S)) ;
#endif
        LAGRAPH_TRY (LAGraph_Free((void *)&Sx, NULL)) ;
        GRB_TRY (GrB_free (&S)) ;
    }

    LG_TRY (LAGraph_Malloc ((void **) &L, n, sizeof (GrB_Index), msg)) ;

    for (GrB_Index i = 0; i < n; i++) {
        L[i] = i ;
    }
    LG_TRY (LAGraph_Malloc ((void **) &L_next, n, sizeof (GrB_Index), msg)) ;

    counts_pool = calloc(max_threads, sizeof(ptable));

    for (int iteration = 0; iteration < itermax; iteration++) {

        int64_t i;
#pragma omp parallel for schedule(dynamic)
        for (i = 0; i < n; i++) {
            #ifdef _OPENMP
            int thread_id = omp_get_thread_num() ;
            #else
            int thread_id = 0 ;
            #endif
            ptable *counts = &counts_pool [thread_id] ;
            GrB_Index* neighbors = Si + Sp[i] ;
            GrB_Index sz = Sp[i+1] - Sp[i] ;
            for (GrB_Index j = 0; j < sz; j++) {
                ptable_add (counts, L[neighbors[j]]) ;
            }
            if (G->kind == LAGraph_ADJACENCY_DIRECTED) {
                neighbors = Ti + Tp[i] ;
                sz = Tp[i+1] - Tp[i] ;
                for (GrB_Index j = 0; j < sz; j++) {
                    ptable_add (counts, L[neighbors[j]]) ;
                }
            }
            GrB_Index best_label, best_count ;
            ptable_reduce (counts, &best_label, &best_count, counts_reducer) ;
            L_next[i] = best_label ;
            ptable_clear (counts) ;
        }

        GrB_Index* tmp = L ; L = L_next ; L_next = tmp ;
        bool changed = false ;
        for (GrB_Index i = 0; i < n; i++) {
            if (L[i] != L_next[i]) {
                changed = true ;
                break ;
            }
        }
        if (!changed) {
             break ;
        }
    }

    ptable_pool_free(counts_pool, max_threads); counts_pool = NULL;

    //--------------------------------------------------------------------------
    // extract final labels to the result vector
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new(&CDLP, GrB_UINT64, n))
    for (GrB_Index i = 0; i < n; i++)
    {
        GrB_Index l = L[i];
//      if (l == GrB_INDEX_MAX + 1) { l = i ; }
        l = (l == GrB_INDEX_MAX + 1) ? i : l ;
        GRB_TRY (GrB_Vector_setElement(CDLP, l, i))
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*CDLP_handle) = CDLP;
    CDLP = NULL;            // set to NULL so LG_FREE_ALL doesn't free it
    LG_FREE_ALL;

    return (GrB_SUCCESS);
}
