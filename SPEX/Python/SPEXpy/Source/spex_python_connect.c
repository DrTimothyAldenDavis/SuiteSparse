//------------------------------------------------------------------------------
// SPEX/Python/SPEXpy/Source/spex_python_connect.c: use SPEX from Python
//------------------------------------------------------------------------------
// SPEX: (c) 2022-2026, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later
//------------------------------------------------------------------------------

#include "spex_python_connect.h"
#include "spex_cholesky_internal.h"
#include <stdlib.h>
#include <string.h>

// Every non-NULL sol_void[i] is allocated with malloc().  Python must copy
// the value, then call this function to release the returned buffer.  Freeing
// inside the same shared library also works with separate C runtimes.
void spex_python_free_output(void *ptr)
{
    free(ptr);
}

// SPEX_CHECK returns immediately, which is inappropriate when both SPEX
// objects and partially constructed Python output buffers need cleanup.
// Keep a single cleanup path instead.
#define PY_CHECK(method)                     \
    do                                       \
    {                                        \
        info = (method);                     \
        if (info != SPEX_OK) goto cleanup;   \
    } while (0)

SPEX_info spex_python
(
    void **sol_void,  // n output pointers; Python frees each non-NULL entry
    int64_t *Ap,      // CSC column pointers (n+1 entries)
    int64_t *Ai,      // CSC row indices (nz entries)
    double *Ax,       // CSC entries (nz entries)
    double *bx,       // dense right-hand side (m entries)
    int m,            // number of rows of A
    int n,            // number of columns of A
    int nz,           // number of stored entries in A
    int ordering,     // Python values: 0=none, 1=COLAMD, 2=AMD, 3=default
    int algorithm,    // 1=automatic, 2=LU, 3=Cholesky, 4=LDL, 5=QR
    bool charOut      // true: rational strings; false: doubles
)
{
    SPEX_info info = SPEX_OK;
    SPEX_matrix A_in = NULL, b_in = NULL, A = NULL, b = NULL;
    SPEX_matrix x = NULL, x2 = NULL;
    SPEX_options option = NULL;
    char *formatted = NULL;
    bool initialized = false;

    // Validate before initializing SPEX; a failed initialization must not be
    // followed by a call to SPEX_finalize().
    if (!sol_void || !Ap || !Ai || !Ax || !bx ||
        m <= 0 || n <= 0 || nz < 0 ||
        ordering < 0 || ordering > 3 || algorithm < 1 || algorithm > 5)
    {
        return SPEX_INCORRECT_INPUT;
    }

    // This also makes partial allocation failure safe to clean up.
    for (int i = 0; i < n; i++)
    {
        sol_void[i] = NULL;
    }

    info = SPEX_initialize();
    if (info != SPEX_OK) return info;
    initialized = true;

    PY_CHECK(SPEX_create_default_options(&option));

    // Do not copy the Python integer directly into the C enum.  Existing
    // Options.py uses 0=none, 1=COLAMD, 2=AMD, unlike the C enum values.
    switch (ordering)
    {
        case 0: option->order = SPEX_NO_ORDERING;      break;
        case 1: option->order = SPEX_COLAMD;           break;
        case 2: option->order = SPEX_AMD;              break;
        case 3: option->order = SPEX_DEFAULT_ORDERING; break;
    }

    // These matrices are shallow: the CSC arrays and RHS belong to NumPy.
    PY_CHECK(SPEX_matrix_allocate(&A_in, SPEX_CSC, SPEX_FP64,
                                  m, n, nz, true, true, option));
    A_in->p = Ap;
    A_in->i = Ai;
    A_in->x.fp64 = Ax;

    // The RHS has m rows, not n.  nzmax for a dense matrix is ignored.
    PY_CHECK(SPEX_matrix_allocate(&b_in, SPEX_DENSE, SPEX_FP64,
                                  m, 1, m, true, true, option));
    b_in->x.fp64 = bx;

    PY_CHECK(SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, A_in, option));
    PY_CHECK(SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ, b_in, option));

    switch (algorithm)
    {
        case 1:
            PY_CHECK(SPEX_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 2:
            PY_CHECK(SPEX_lu_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 3:
            PY_CHECK(SPEX_cholesky_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 4:
            PY_CHECK(SPEX_ldl_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 5:
            PY_CHECK(SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option));
            break;
    }

    if (x == NULL || x->kind != SPEX_DENSE || x->type != SPEX_MPQ ||
        x->m != n || x->n != 1)
    {
        info = SPEX_INCORRECT_INPUT;
        goto cleanup;
    }

    // Allocate output buffers independently of SPEX's matrices.  On any
    // failure, cleanup frees all buffers already created.
    if (charOut)
    {
        for (int i = 0; i < n; i++)
        {
            PY_CHECK(SPEX_mpfr_asprintf(&formatted, "%Qd", x->x.mpq[i]));
            if (formatted == NULL)
            {
                info = SPEX_OUT_OF_MEMORY;
                goto cleanup;
            }
            size_t length = strlen(formatted) + 1;
            sol_void[i] = malloc(length);
            if (sol_void[i] == NULL)
            {
                info = SPEX_OUT_OF_MEMORY;
                goto cleanup;
            }
            memcpy(sol_void[i], formatted, length);
            SPEX_mpfr_free_str(formatted);
            formatted = NULL;
        }
    }
    else
    {
        PY_CHECK(SPEX_matrix_copy(&x2, SPEX_DENSE, SPEX_FP64, x, option));
        for (int i = 0; i < n; i++)
        {
            sol_void[i] = malloc(sizeof(double));
            if (sol_void[i] == NULL)
            {
                info = SPEX_OUT_OF_MEMORY;
                goto cleanup;
            }
            *((double *) sol_void[i]) = x2->x.fp64[i];
        }
    }

cleanup:
    if (formatted != NULL)
    {
        SPEX_mpfr_free_str(formatted);
    }

    if (info != SPEX_OK && sol_void != NULL)
    {
        for (int i = 0; i < n; i++)
        {
            free(sol_void[i]);
            sol_void[i] = NULL;
        }
    }

    // Free the borrowed-array headers (not the NumPy-owned arrays), exact
    // matrices, converted output, options, and initialized SPEX environment.
    if (initialized)
    {
        SPEX_matrix_free(&x2, option);
        SPEX_matrix_free(&x, option);
        SPEX_matrix_free(&b, option);
        SPEX_matrix_free(&A, option);
        SPEX_matrix_free(&b_in, option);
        SPEX_matrix_free(&A_in, option);
        SPEX_FREE(option);
        SPEX_finalize();
    }
    return info;
}

SPEX_info spex_python_rank
(
    int64_t *rank_out,
    int64_t *Ap,
    int64_t *Ai,
    double *Ax,
    int m,
    int n,
    int nz
)
{
    SPEX_info info = SPEX_OK;
    SPEX_matrix A_in = NULL, A = NULL;
    SPEX_options option = NULL;
    bool initialized = false;

    if (!rank_out || !Ap || !Ai || !Ax || m <= 0 || n <= 0 || nz < 0)
    {
        return SPEX_INCORRECT_INPUT;
    }

    info = SPEX_initialize();
    if (info != SPEX_OK) return info;
    initialized = true;

    PY_CHECK(SPEX_create_default_options(&option));
    PY_CHECK(SPEX_matrix_allocate(&A_in, SPEX_CSC, SPEX_FP64,
                                  m, n, nz, true, true, option));
    A_in->p = Ap;
    A_in->i = Ai;
    A_in->x.fp64 = Ax;

    PY_CHECK(SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, A_in, option));
    PY_CHECK(SPEX_rank(rank_out, A, option));

cleanup:
    if (initialized)
    {
        SPEX_matrix_free(&A, option);
        SPEX_matrix_free(&A_in, option);
        SPEX_FREE(option);
        SPEX_finalize();
    }
    return info;
}
