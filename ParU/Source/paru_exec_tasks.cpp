////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_exec_tasks.cpp ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  execute a set of tasks for numeric factorization.
 *
 * @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// paru_exec_tasks_seq: execute all tasks on a single thread
//------------------------------------------------------------------------------

ParU_Info paru_exec_tasks_seq
(
    int64_t t,
    int64_t *task_num_child,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
)
{

    DEBUGLEVEL(0);
    const int64_t *task_parent = Sym->task_parent;
    int64_t daddy = task_parent[t];
    const int64_t *task_map = Sym->task_map;

    int64_t num_original_children = 0;
    if (daddy != -1)
    {
        num_original_children = Sym->task_num_child[daddy];
    }
    PRLEVEL(1, ("Seq: executing task " LD " fronts " LD "-" LD " (" LD " children)\n", t,
                task_map[t] + 1, task_map[t + 1], num_original_children));
    ParU_Info myInfo;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    for (int64_t f = task_map[t] + 1; f <= task_map[t + 1]; f++)
    {
        PRLEVEL(2, ("Seq: calling " LD "\n", f));
        myInfo = paru_front(f, Work, Sym, Num);
        if (myInfo != PARU_SUCCESS)
        {
            return myInfo;
        }
    }
    int64_t num_rem_children;
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("task time task " LD " is %lf\n", t, time));
#endif

#ifndef NDEBUG
    if (daddy == -1) PRLEVEL(1, ("%% finished task root(" LD ")\n", t));
#endif

    if (daddy != -1)  // if it is not a root
    {
        if (num_original_children != 1)
        {
            task_num_child[daddy]--;
            num_rem_children = task_num_child[daddy];

            PRLEVEL(1,
                    ("%%Seq finished task " LD "(" LD "," LD ")Parent has " LD " left\n", t,
                     task_map[t] + 1, task_map[t + 1], task_num_child[daddy]));
            if (num_rem_children == 0)
            {
                PRLEVEL(
                    1, ("%%Seq task " LD " executing its parent " LD "\n", t, daddy));
                return myInfo = paru_exec_tasks_seq(daddy, task_num_child, Work,
                                                    Sym, Num);
            }
        }
        else  // I was the only spoiled kid in the family;
        {
            PRLEVEL(1, ("%% Seq task " LD " only child executing its parent " LD "\n",
                        t, daddy));
            return myInfo = paru_exec_tasks_seq(daddy, task_num_child, Work,
                        Sym, Num);
        }
    }
    return myInfo;
}

//------------------------------------------------------------------------------
// paru_exec_tasks: execute all tasks in parallel
//------------------------------------------------------------------------------

ParU_Info paru_exec_tasks
(
    int64_t t,
    int64_t *task_num_child,
    int64_t &chain_task,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
)
{

    const int64_t *task_parent = Sym->task_parent;
    int64_t daddy = task_parent[t];
    const int64_t *task_map = Sym->task_map;

    int64_t num_original_children = 0;
    if (daddy != -1)
    {
        num_original_children = Sym->task_num_child[daddy];
    }
    PRLEVEL(1, ("executing task " LD " fronts " LD "-" LD " (" LD " children)\n", t,
                task_map[t] + 1, task_map[t + 1], num_original_children));
    ParU_Info myInfo;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    for (int64_t f = task_map[t] + 1; f <= task_map[t + 1]; f++)
    {
        myInfo = paru_front(f, Work, Sym, Num);
        if (myInfo != PARU_SUCCESS) return myInfo;
    }
    int64_t num_rem_children;
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("task time task " LD " is %lf\n", t, time));
#endif

#ifndef NDEBUG
    if (daddy == -1) PRLEVEL(1, ("%% finished task root(" LD ")\n", t));
#endif

    if (daddy != -1)  // if it is not a root
    {
        if (num_original_children != 1)
        {
            #pragma omp atomic capture
            {
                task_num_child[daddy]--;
                num_rem_children = task_num_child[daddy];
            }

            PRLEVEL(1,
                    ("%% finished task " LD "(" LD "," LD ")  Parent has " LD " left\n", t,
                     task_map[t] + 1, task_map[t + 1], task_num_child[daddy]));
            if (num_rem_children == 0)
            {
                PRLEVEL(1,
                        ("%% task " LD " executing its parent " LD "\n", t, daddy));
                int64_t resq;
                #pragma omp atomic read
                resq = Work->resq;
                if (resq == 1)
                {
                    chain_task = daddy;
                    PRLEVEL(2, ("%% CHAIN ALERT1: task " LD " calling " LD ""
                                " resq = " LD "\n",
                                t, daddy, resq));
                }
                else
                {
                    return myInfo = paru_exec_tasks(daddy, task_num_child,
                         chain_task, Work, Sym, Num);
                }
            }
        }
        else  // I was the only spoiled kid in the family;
        {
            PRLEVEL(1, ("%% task " LD " only child executing its parent " LD "\n", t,
                        daddy));
            int64_t resq;
            #pragma omp atomic read
            resq = Work->resq;

            if (resq == 1)
            {
                chain_task = daddy;
                PRLEVEL(2, ("%% CHAIN ALERT1: task " LD " calling " LD ""
                            " resq = " LD "\n",
                            t, daddy, resq));
            }
            else
            {
                return myInfo = paru_exec_tasks(daddy, task_num_child,
                    chain_task, Work, Sym, Num);
            }
        }
    }
    return myInfo;
}

