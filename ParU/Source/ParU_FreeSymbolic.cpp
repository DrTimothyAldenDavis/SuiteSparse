////////////////////////////////////////////////////////////////////////////////
////////////////////////// ParU_FreeSymbolic.cpp ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  Free the Symbolic analysis object.
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_FreeSymbolic: free the symbolic analysis data structure
//------------------------------------------------------------------------------

ParU_Info ParU_FreeSymbolic
(
    // input/output:
    ParU_Symbolic **Sym_handle, // symbolic object to free
    // control:
    ParU_Control *Control
)
{
    if (Sym_handle == NULL || *Sym_handle == NULL)
    {
        // nothing to do
        return PARU_SUCCESS;
    }
    if (!Control)
    {
        return (PARU_INVALID) ;
    }

    DEBUGLEVEL(0);
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
    {
        // freeing singletons
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

