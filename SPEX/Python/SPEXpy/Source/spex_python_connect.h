//------------------------------------------------------------------------------
// SPEX/Python/spex_connect.h: use SPEX in Python
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "SPEX.h"

SPEX_info spex_python
(
     //output
     void **sol_void, // solution
     //input
     int64_t *Ap,     // column pointers of A, an array size is n+1
     int64_t *Ai,     // row indices of A, of size nzmax.
     double *Ax,      // values of A
     double *bx,      // values of b
     int m,           // Number of rows of A
     int n,           // Number of columns of A
     int nz,          // Number of nonzeros in A
     int ordering,    // type of ordering: 0-none, 1-colamd, 2-amd
     int algorithm,   //  1-backslash, 2-left lu, 3-cholesky
     bool charOut     // True if char ** output, false if double
);

