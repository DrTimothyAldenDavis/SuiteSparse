////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_bin_search ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief binary search in different contexts.
 *  IMPORTANT: it includes r.
 *
 * @author Aznaveh
 *  */
#include "paru_internal.hpp"
#ifdef PARU_COVERAGE
#define LEN 8
#else
#define LEN 0
#endif

int64_t paru_bin_srch(int64_t *srt_lst, int64_t l, int64_t r, int64_t num)
// a simple binary search for when we know all the indices are available
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% BINSearch " LD "," LD " for " LD "\n", l, r, num));
    if (r >= l + LEN)
    {
        int64_t mid = l + (r - l) / 2;
        PRLEVEL(1, ("%% mid is " LD "\n", mid));
        if (srt_lst[mid] == num) return mid;

        if (srt_lst[mid] > num)
        {
            PRLEVEL(1, ("%% 1 New " LD "," LD " \n", l, mid - 1));
            return paru_bin_srch(srt_lst, l, mid - 1, num);
        }
        PRLEVEL(1, ("%% 2 New " LD "," LD " \n", mid + 1, r));
        return paru_bin_srch(srt_lst, mid + 1, r, num);
    }

    if (r >= l)
    {
        for (int64_t i = l; i <= r; i++)
            if (srt_lst[i] == num) return i;
    }

    return (-1);
}

int64_t paru_bin_srch_col(int64_t *srt_lst, int64_t l, int64_t r, int64_t num)
// a simple binary search for when it is possible that some columns were
// flipped
{
    if (r >= l + LEN)
    {
        int64_t mid = l + (r - l) / 2;
        int64_t srt_lstMid = (srt_lst[mid] < 0) ? flip(srt_lst[mid]) : srt_lst[mid];
        if (srt_lstMid == num) return mid;

        if (srt_lstMid > num) return paru_bin_srch_col(srt_lst, l, mid - 1, num);
        return paru_bin_srch_col(srt_lst, mid + 1, r, num);
    }

    if (r >= l)
    {
        for (int64_t i = l; i <= r; i++)
            if (srt_lst[i] == num) return i;
    }
    return (-1);
}
