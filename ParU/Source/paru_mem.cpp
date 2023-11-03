////////////////////////////////////////////////////////////////////////////////
////////////////////////// paru_mem.cpp ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  Wrappers for managing memory
 *  allocating and freeing is done through SuiteSparse and these wrappers
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

#ifdef PARU_ALLOC_TESTING
// global variables
bool paru_malloc_tracking = false;
int64_t paru_nmalloc = 0;


bool paru_get_malloc_tracking(void)
{
    bool track;
#pragma omp critical (paru_malloc_testing)
    {
        track = paru_malloc_tracking;
    }
    return (track);
}

void paru_set_malloc_tracking(bool track)
{
#pragma omp critical (paru_malloc_testing)
    {
        paru_malloc_tracking = track;
    }
}

void paru_set_nmalloc(int64_t nmalloc)
{
#pragma omp critical (paru_malloc_testing)
    {
        paru_nmalloc = nmalloc;
    }
    //printf("inside set nmalloc=" LD "",nmalloc);
    PRLEVEL(1, ("inside set nmalloc=" LD "",nmalloc));
}

int64_t paru_decr_nmalloc(void)
{
    int64_t nmalloc = 0;
#pragma omp critical (paru_malloc_testing)
    {
        if (paru_nmalloc > 0)
        {
            nmalloc = paru_nmalloc--;
        }
    }
    //printf("inside decr nmalloc=" LD "",nmalloc);
    PRLEVEL(1, ("inside decr nmalloc=" LD "",nmalloc));
    return (nmalloc);
}

int64_t paru_get_nmalloc(void)
{
    int64_t nmalloc = 0;
#pragma omp critical (paru_malloc_testing)
    {
        nmalloc = paru_nmalloc;
    }
    //printf("inside get nmalloc=" LD "",nmalloc);
    PRLEVEL(1, ("inside get nmalloc=" LD "",nmalloc));
    return (nmalloc);
}

#endif

//  Wrapper around malloc routine
//
//  Uses a pointer to the malloc routine.
void *paru_alloc(size_t n, size_t size)
{
    DEBUGLEVEL(0);
#ifndef NDEBUG
    static int64_t alloc_count = 0;
#endif
    void *p = NULL;
    if (size == 0)
    {
        PRLEVEL(1, ("ParU: size must be > 0\n"));
        return NULL;
    }
    else if (n >= (Size_max / size) || n >= INT_MAX)
    {
        // object is too big to allocate without causing integer overflow
        PRLEVEL(1, ("ParU: problem too large\n"));
        p = NULL;
    }
    else
    {
#ifdef PARU_ALLOC_TESTING
        // brutal memory testing only
        if (paru_get_malloc_tracking())
        {
            int64_t nmalloc = paru_decr_nmalloc();
            if (nmalloc > 0)
            {
                p = SuiteSparse_malloc(n, size);
            }
        }
        else
        {
            p = SuiteSparse_malloc(n, size);
        }
#else
        // in production
        p = SuiteSparse_malloc(n, size);
#endif

        if (p == NULL)
        {
            // out of memory
            PRLEVEL(1, ("ParU: out of memory\n"));
        }
        else
        {
#ifndef NDEBUG
            PRLEVEL(1, ("%% allocated " LD " in %p total= " LD "\n", n * size, p,
                        alloc_count));
            alloc_count += n * size;
#endif
        }
    }
    return p;
}

//  Wrapper around calloc routine
//
//  Uses a pointer to the calloc routine.
void *paru_calloc(size_t n, size_t size)
{
    DEBUGLEVEL(0);
#ifndef NDEBUG
    static int64_t calloc_count = 0;
#endif
    void *p = NULL;
    if (size == 0)
    {
        PRLEVEL(1, ("ParU: size must be > 0\n"));
    }
    else if (n >= (Size_max / size) || n >= INT_MAX)
    {
        // object is too big to allocate without causing integer overflow
        PRLEVEL(1, ("ParU: problem too large\n"));
    }
    else
    {
#ifdef PARU_ALLOC_TESTING
        // brutal memory testing only
        if (paru_get_malloc_tracking())
        {
            int64_t nmalloc = paru_decr_nmalloc();
            if (nmalloc > 0)
            {
                p = SuiteSparse_calloc(n, size);
            }
        }
        else
        {
            p = SuiteSparse_calloc(n, size);
        }
#else
        // in production
        p = SuiteSparse_calloc(n, size);
#endif

        if (p == NULL)
        {
            // out of memory
            PRLEVEL(1, ("ParU: out of memory\n"));
        }
        else
        {
#ifndef NDEBUG
            PRLEVEL(1, ("%% callocated " LD " in %p total= " LD "\n", n * size, p,
                        calloc_count));
            calloc_count += n * size;
#endif
        }
    }
    return p;
}

//  Wrapper around realloc routine
//
//  Uses a pointer to the realloc routine.
void *paru_realloc(
    size_t nnew,     // requested # of items
    size_t size_Entry,  // size of each Entry
    void *p,         // block memory to realloc
    size_t *n)       // current size on input, nnew output if successful 
{
    DEBUGLEVEL(0);
#ifndef NDEBUG
    static int64_t realloc_count = 0;
#endif
    void *pnew;
    if (size_Entry == 0)
    {
        PRLEVEL(1, ("ParU: sizeof(entry)  must be > 0\n"));
        p = NULL;
    }
    else if (p == NULL)
    {  // A new alloc
        p = paru_alloc(nnew, size_Entry);
        *n = (p == NULL) ? 0 : nnew;
    }
    else if (nnew == *n )
    {
        PRLEVEL(1, ("%% reallocating nothing " LD ", " LD " in %p \n", nnew, *n,
                    p));
    }
    else if (nnew >= (Size_max / size_Entry) || nnew >= INT_MAX)
    {
        // object is too big to allocate without causing integer overflow
        PRLEVEL(1, ("ParU: problem too large\n"));
    }

    else
    {  // The object exists, and is changing to some other nonzero size.
        PRLEVEL(1, ("realloc : " LD " to " LD ", " LD "\n", *n, nnew, size_Entry));
        int ok = TRUE;

#ifdef PARU_ALLOC_TESTING
        // brutal memory testing only
        if (paru_get_malloc_tracking())
        {
            int64_t nmalloc = paru_decr_nmalloc();
            if (nmalloc > 0)
            {
                pnew = SuiteSparse_realloc(nnew, *n, size_Entry, p, &ok);
            }
            else
            {
                // pretend to fail
                ok = FALSE;
            }
        }
        else
        {
            pnew = SuiteSparse_realloc(nnew, *n, size_Entry, p, &ok);
        }
#else
        // in production
        pnew = SuiteSparse_realloc(nnew, *n, size_Entry, p, &ok);
#endif

        if (ok)
        {
#ifndef NDEBUG
            realloc_count += nnew * size_Entry - *n;
            PRLEVEL(1, ("%% reallocated " LD " in %p and freed %p total= " LD "\n",
                        nnew* size_Entry, pnew, p, realloc_count));
#endif
	    p = pnew ;
        *n = nnew;
        }
    }
    return p;
}

//  Wrapper around free routine
//
void paru_free(size_t n, size_t size, void *p)
{
    DEBUGLEVEL(0);

    // static int64_t free_count = 0;
    // free_count += n * size;

    // Valgrind is unhappy about some part here
    //    PRLEVEL (1, ("%% free " LD " in %p total= " LD "\n",
    //                n*size, p, free_count));

    if (p != NULL)
        SuiteSparse_free(p);
    else
    {
        PRLEVEL(1, ("%% freeing a NULL pointer  \n"));
    }
}

//  Global replacement of new and delete
//
void *operator new(size_t size)
{  // no inline, required by [replacement.functions]/3
    DEBUGLEVEL(0);
#ifndef NDEBUG
    static int64_t cpp_count = 0;
    cpp_count += size;
    PRLEVEL(1, ("global op new called, size = %zu tot=" LD "\n", size, cpp_count));
#endif

    if (size == 0)
        ++size;  // avoid malloc(0) which may return nullptr on success

    if (void *ptr = paru_alloc(1, size)) return ptr;
    throw std::bad_alloc{};
}

void operator delete(void *ptr) noexcept
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("global op delete called"));
    paru_free(0, 0, ptr);
}

//  freeing symbolic analysis data structure
ParU_Ret ParU_Freesym(ParU_Symbolic **Sym_handle, ParU_Control *Control)
{
    DEBUGLEVEL(0);
    if (Sym_handle == NULL || *Sym_handle == NULL)
        // nothing to do
        return PARU_SUCCESS;

    ParU_Symbolic *Sym;
    Sym = *Sym_handle;

    int64_t m = Sym->m;
    int64_t n = Sym->n;
    int64_t n1 = Sym->n1;
    int64_t nf = Sym->nf;
    int64_t snz = Sym->snz;
    PRLEVEL(1, ("%% In free sym: m=" LD " n=" LD "\n nf=" LD " "
                "Sym->anz=" LD " \n",
                m, n, nf, Sym->anz));

    paru_free(nf + 1, sizeof(int64_t), Sym->Parent);
    paru_free(nf + 1, sizeof(int64_t), Sym->Child);
    paru_free(nf + 2, sizeof(int64_t), Sym->Childp);
    paru_free(nf + 1, sizeof(int64_t), Sym->Super);
    paru_free(nf, sizeof(int64_t), Sym->Depth);
    paru_free(n, sizeof(int64_t), Sym->Qfill);
    paru_free(n, sizeof(int64_t), Sym->Diag_map);
    paru_free((m + 1), sizeof(int64_t), Sym->Pinit);
    paru_free(nf + 1, sizeof(int64_t), Sym->Fm);
    paru_free(nf + 1, sizeof(int64_t), Sym->Cm);

    // paru_free(Sym->num_roots, sizeof(int64_t), Sym->roots);

    paru_free(m + 1 - n1, sizeof(int64_t), Sym->Sp);
    paru_free(snz, sizeof(int64_t), Sym->Sj);
    paru_free(n + 2 - n1, sizeof(int64_t), Sym->Sleft);

    // paru_free((n + 1), sizeof(int64_t), Sym->Chain_start);
    // paru_free((n + 1), sizeof(int64_t), Sym->Chain_maxrows);
    // paru_free((n + 1), sizeof(int64_t), Sym->Chain_maxcols);

    paru_free(nf + 1, sizeof(double), Sym->front_flop_bound);
    paru_free(nf + 1, sizeof(double), Sym->stree_flop_bound);

    int64_t ms = m - n1;  // submatrix is msxns

    paru_free(ms + nf, sizeof(int64_t), Sym->aParent);
    paru_free(ms + nf + 1, sizeof(int64_t), Sym->aChild);
    paru_free(ms + nf + 2, sizeof(int64_t), Sym->aChildp);
    paru_free(ms, sizeof(int64_t), Sym->row2atree);
    paru_free(nf, sizeof(int64_t), Sym->super2atree);
    paru_free(nf + 1, sizeof(int64_t), Sym->first);
    paru_free(m, sizeof(int64_t), Sym->Pinv);

    if (n1 > 0)
    {  // freeing singletons
        int64_t cs1 = Sym->cs1;
        if (cs1 > 0)
        {
            ParU_U_singleton ustons = Sym->ustons;
            paru_free(cs1 + 1, sizeof(int64_t), ustons.Sup);
            int64_t nnz = ustons.nnz;
            paru_free(nnz, sizeof(int64_t), ustons.Suj);
        }

        int64_t rs1 = Sym->rs1;
        if (rs1 > 0)
        {
            ParU_L_singleton lstons = Sym->lstons;
            paru_free(rs1 + 1, sizeof(int64_t), lstons.Slp);
            int64_t nnz = lstons.nnz;
            paru_free(nnz, sizeof(int64_t), lstons.Sli);
        }
    }
    int64_t ntasks = Sym->ntasks;
    paru_free(ntasks + 1, sizeof(int64_t), Sym->task_map);
    paru_free(ntasks, sizeof(int64_t), Sym->task_parent);
    paru_free(ntasks, sizeof(int64_t), Sym->task_num_child);
    paru_free(ntasks, sizeof(int64_t), Sym->task_depth);

    paru_free(1, sizeof(ParU_Symbolic), Sym);

    *Sym_handle = NULL;
    return PARU_SUCCESS;
}

// free element e from elementList
void paru_free_el(int64_t e, paru_element **elementList)
{
    DEBUGLEVEL(0);
    paru_element *el = elementList[e];
    if (el == NULL) return;
#ifndef NDEBUG
    int64_t nrows = el->nrows, ncols = el->ncols;
    PRLEVEL(1, ("%%Free the element e =" LD "\t", e));
    PRLEVEL(1, ("%% nrows =" LD " ", nrows));
    PRLEVEL(1, ("%% ncols =" LD "\n", ncols));
    int64_t tot_size = 
        sizeof(paru_element) + sizeof(int64_t)
        * (2 * (nrows + ncols)) + sizeof(double) * nrows * ncols;
    paru_free(1, tot_size, el);
#else
    paru_free(1, 0, el);
#endif
    elementList[e] = NULL;
}

ParU_Ret paru_free_work(ParU_Symbolic *Sym, paru_work *Work)
{
    int64_t m = Sym->m - Sym->n1;
    int64_t nf = Sym->nf;
    int64_t n = Sym->n - Sym->n1;
    paru_free(m, sizeof(int64_t), Work->rowSize);
    paru_free(m + nf + 1, sizeof(int64_t), Work->rowMark);
    paru_free(m + nf, sizeof(int64_t), Work->elRow);
    paru_free(m + nf, sizeof(int64_t), Work->elCol);
    paru_free(Sym->ntasks, sizeof(int64_t), Work->task_num_child);

    paru_free(1, nf * sizeof(int64_t), Work->time_stamp);

    paru_tupleList *RowList = Work->RowList;
    PRLEVEL(1, ("%% RowList =%p\n", RowList));

    if (RowList)
    {
        for (int64_t row = 0; row < m; row++)
        {
            int64_t len = RowList[row].len;
            paru_free(len, sizeof(paru_tuple), RowList[row].list);
        }
    }
    paru_free(1, m * sizeof(paru_tupleList), RowList);

    if (Work->Diag_map)
    {
        paru_free(n, sizeof(int64_t), Work->Diag_map);
        paru_free(n, sizeof(int64_t), Work->inv_Diag_map);
    }

    paru_element **elementList;
    elementList = Work->elementList;

    PRLEVEL(1, ("%% Sym = %p\n", Sym));
    PRLEVEL(1, ("%% freeing initialized elements:\n"));
    if (elementList)
    {
        for (int64_t i = 0; i < m; i++)
        {                               // freeing all row elements
            int64_t e = Sym->row2atree[i];  // element number in augmented tree
            PRLEVEL(1, ("%% e =" LD "\t", e));
            paru_free_el(e, elementList);
        }

        PRLEVEL(1, ("\n%% freeing CB elements:\n"));
        for (int64_t i = 0; i < nf; i++)
        {                                 // freeing all other elements
            int64_t e = Sym->super2atree[i];  //element number in augmented tree
            paru_free_el(e, elementList);
        }
    }

    paru_free(1, (m + nf + 1) * sizeof(paru_element), elementList);

    paru_free(m + nf, sizeof(int64_t), Work->lacList);

    // in practice each parent should deal with the memory for the children
    std::vector<int64_t> **heapList = Work->heapList;
    // freeing memory of heaps.
    if (heapList != NULL)
    {
        for (int64_t eli = 0; eli < m + nf + 1; eli++)
        {
            if (heapList[eli] != NULL)
            {
                PRLEVEL(1,
                        ("%% " LD " has not been freed %p\n", eli, heapList[eli]));
                delete heapList[eli];
                heapList[eli] = NULL;
            }
            ASSERT(heapList[eli] == NULL);
        }
    }
    paru_free(1, (m + nf + 1)*sizeof(std::vector<int64_t> **), Work->heapList);
    paru_free(m, sizeof(int64_t), Work->row_degree_bound);

    return PARU_SUCCESS;
}

ParU_Ret ParU_Freenum(ParU_Numeric **Num_handle, ParU_Control *Control)
{
    DEBUGLEVEL(0);
    if (Num_handle == NULL || *Num_handle == NULL)
    {
        // nothing to do
        return PARU_SUCCESS;
    }

    ParU_Numeric *Num;
    Num = *Num_handle;

    int64_t nf = Num->nf;

    // freeing the numerical input
    paru_free(Num->snz, sizeof(double), Num->Sx);
    if (Num->sunz > 0)
    {
        paru_free(Num->sunz, sizeof(double), Num->Sux);
    }
    if (Num->slnz > 0)
    {
        paru_free(Num->slnz, sizeof(double), Num->Slx);
    }

    paru_free(Num->sym_m, sizeof(int64_t), Num->Rs);  
    paru_free(Num->sym_m, sizeof(int64_t), Num->Pfin);
    paru_free(Num->sym_m, sizeof(int64_t), Num->Ps);

    // free the factors
    ParU_Factors *LUs = Num->partial_LUs;
    ParU_Factors *Us = Num->partial_Us;

    for (int64_t i = 0; i < nf; i++)
    {
        if (Num->frowList)
            paru_free(Num->frowCount[i], sizeof(int64_t), Num->frowList[i]);
        if (Num->fcolList)
            paru_free(Num->fcolCount[i], sizeof(int64_t), Num->fcolList[i]);

        if (Us)
        {
            if (Us[i].p != NULL)
            {
                PRLEVEL(1, ("%% Freeing Us=%p\n", Us[i].p));
                int64_t mm = Us[i].m;
                int64_t nn = Us[i].n;
                paru_free(mm * nn, sizeof(double), Us[i].p);
            }
        }

        if (LUs)
        {
            if (LUs[i].p != NULL)
            {
                PRLEVEL(1, ("%% Freeing LUs=%p\n", LUs[i].p));
                int64_t mm = LUs[i].m;
                int64_t nn = LUs[i].n;
                paru_free(mm * nn, sizeof(double), LUs[i].p);
            }
        }
    }

    PRLEVEL(1, ("%% Done LUs\n"));
    paru_free(1, nf * sizeof(int64_t), Num->frowCount);
    paru_free(1, nf * sizeof(int64_t), Num->fcolCount);

    paru_free(1, nf * sizeof(int64_t *), Num->frowList);
    paru_free(1, nf * sizeof(int64_t *), Num->fcolList);

    paru_free(1, nf * sizeof(ParU_Factors), LUs);
    paru_free(1, nf * sizeof(ParU_Factors), Us);

    paru_free(1, sizeof(ParU_Numeric), Num);
    *Num_handle = NULL;
    return PARU_SUCCESS;
}
