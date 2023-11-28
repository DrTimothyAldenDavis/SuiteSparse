////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_init_rel /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief Initializing element f's  time_stamp
 *          check all f's children and find the maximum time_stamp
 *          this time is checking for validation or invalidation of elements
 *
 * @param Num: pointer to matrix info
 *        f: front that is going to be initialized;
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"
void paru_init_rel(int64_t f, paru_work *Work)
{
    DEBUGLEVEL(0);
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *time_stamp = Work->time_stamp;

    int64_t *Child = Sym->Child;
    int64_t *Childp = Sym->Childp;
    int64_t max_time = 0;

    PRLEVEL(1, ("%% begining=" LD " end=" LD " \n", Childp[f], Childp[f + 1]));
    PRLEVEL(1, ("%% children of " LD "  are:\n", f));
    for (int64_t p = Childp[f]; p <= Childp[f + 1] - 1; p++)
    {
        int64_t child_rel;
        ASSERT(Child[p] >= 0);
        child_rel = time_stamp[Child[p]];
        PRLEVEL(1, ("%% Child[" LD "]= " LD "  ", p, Child[p]));
        max_time = max_time > child_rel ? max_time : child_rel;
    }
    time_stamp[f] = ++max_time;
    PRLEVEL(1, ("%% max_time=" LD " \n", max_time));
}
