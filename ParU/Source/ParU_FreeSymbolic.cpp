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
    ParU_Symbolic *Sym_handle, // symbolic object to free
    // control:
    ParU_Control Control
)
{
    if (Sym_handle == NULL || *Sym_handle == NULL)
    {
        // nothing to do
        return PARU_SUCCESS;
    }

    DEBUGLEVEL(0);
    ParU_Symbolic Sym ;
    Sym = *Sym_handle;

    int64_t m = Sym->m;
    int64_t n = Sym->n;
    int64_t n1 = Sym->n1;
    int64_t nf = Sym->nf;
    int64_t snz = Sym->snz;
    PRLEVEL(1, ("%% In free sym: m=" LD " n=" LD "\n nf=" LD " "
                "Sym->anz=" LD " \n",
                m, n, nf, Sym->anz));

    PARU_FREE(nf + 1, int64_t, Sym->Parent);
    PARU_FREE(nf + 1, int64_t, Sym->Child);
    PARU_FREE(nf + 2, int64_t, Sym->Childp);
    PARU_FREE(nf + 1, int64_t, Sym->Super);
    PARU_FREE(nf, int64_t, Sym->Depth);
    PARU_FREE(n, int64_t, Sym->Qfill);
    PARU_FREE(n, int64_t, Sym->Diag_map);
    PARU_FREE(m + 1, int64_t, Sym->Pinit);
    PARU_FREE(nf + 1, int64_t, Sym->Fm);
    PARU_FREE(nf + 1, int64_t, Sym->Cm);
    PARU_FREE(m + 1 - n1, int64_t, Sym->Sp);
    PARU_FREE(snz, int64_t, Sym->Sj);
    PARU_FREE(n + 2 - n1, int64_t, Sym->Sleft);
    PARU_FREE(nf + 1, double, Sym->front_flop_bound);
    PARU_FREE(nf + 1, double, Sym->stree_flop_bound);

    int64_t ms = m - n1;  // submatrix is msxns

    PARU_FREE(ms + nf, int64_t, Sym->aParent);
    PARU_FREE(ms + nf + 1, int64_t, Sym->aChild);
    PARU_FREE(ms + nf + 2, int64_t, Sym->aChildp);
    PARU_FREE(ms, int64_t, Sym->row2atree);
    PARU_FREE(nf, int64_t, Sym->super2atree);
    PARU_FREE(nf + 1, int64_t, Sym->first);
    PARU_FREE(m, int64_t, Sym->Pinv);

    if (n1 > 0)
    {
        // freeing singletons
        int64_t cs1 = Sym->cs1;
        if (cs1 > 0)
        {
            ParU_U_singleton ustons = Sym->ustons;
            PARU_FREE(cs1 + 1, int64_t, ustons.Sup);
            int64_t nnz = ustons.nnz;
            PARU_FREE(nnz, int64_t, ustons.Suj);
        }

        int64_t rs1 = Sym->rs1;
        if (rs1 > 0)
        {
            ParU_L_singleton lstons = Sym->lstons;
            PARU_FREE(rs1 + 1, int64_t, lstons.Slp);
            int64_t nnz = lstons.nnz;
            PARU_FREE(nnz, int64_t, lstons.Sli);
        }
    }
    int64_t ntasks = Sym->ntasks;
    PARU_FREE(ntasks + 1, int64_t, Sym->task_map);
    PARU_FREE(ntasks, int64_t, Sym->task_parent);
    PARU_FREE(ntasks, int64_t, Sym->task_num_child);
    PARU_FREE(ntasks, int64_t, Sym->task_depth);
    PARU_FREE(1, ParU_Symbolic_struct, Sym);
    (*Sym_handle) = NULL;
    return (PARU_SUCCESS) ;
}

