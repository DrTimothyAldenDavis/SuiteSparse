////////////////////////////////////////////////////////////////////////////////
/////////////////////////// paru_Diag_update ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  updating the diagonal map when strategy is symmetric but for some
 *      reason the diagonal entry is not picked
 *
 *                  \o\     col2
 *                    \\    |
 *                      \\  |   Instead of picking o we are picking x
 *           new_d  x-----\\|    so we put x's row for col2
 *                          \\
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"

void paru_Diag_update(int64_t pivcol, int64_t pivrow, paru_work *Work)

{
    DEBUGLEVEL(0);
    int64_t *Diag_map = Work->Diag_map;
    int64_t *inv_Diag_map = Work->inv_Diag_map;

    ASSERT(Diag_map);
    ASSERT(inv_Diag_map);

    int64_t diag_row = Diag_map[pivcol];

    Diag_map[pivcol] = pivrow;
    int64_t col2 = inv_Diag_map[pivrow];
    Diag_map[col2] = diag_row;

    PRLEVEL(1, ("%% Inside Diag update pivcol=" LD " pivrow=" LD ""
                " diag_row=" LD " col2=" LD "\n",
                pivcol, pivrow, diag_row, col2));

    inv_Diag_map[diag_row] = col2;
    inv_Diag_map[pivrow] = pivcol;
}
