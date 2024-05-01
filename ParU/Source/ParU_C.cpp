////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// ParU_C.cpp ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//
/*! @brief  This C++ file provides a set of C-callable wrappers so that a C
 *  program can call ParU.
 *
 *  @author Aznaveh
 */

#include "paru_internal.hpp"

extern "C"
{

//------------------------------------------------------------------------------
// ParU_Version:
//------------------------------------------------------------------------------

// return the version
ParU_Info ParU_C_Version (int ver [3], char date [128])
{
    return (ParU_Version (ver, date)) ;
}

//------------------------------------------------------------------------------
// ParU_C_Init_Control: initialize C_Control with the default values
//------------------------------------------------------------------------------

ParU_Info ParU_C_Init_Control (ParU_C_Control *Control_C)
{
    if (!Control_C)
    {
        return (PARU_INVALID) ;
    }

    Control_C->mem_chunk = PARU_MEM_CHUNK ;
    Control_C->umfpack_ordering = UMFPACK_ORDERING_METIS ;
    Control_C->umfpack_strategy = UMFPACK_STRATEGY_AUTO ;
    Control_C->filter_singletons = 1 ;
    Control_C->relaxed_amalgamation = 32 ;
    Control_C->prescale = 1 ;
    Control_C->panel_width = 32 ;
    Control_C->paru_strategy = PARU_STRATEGY_AUTO ;
    Control_C->piv_toler = .1 ;
    Control_C->diag_toler = .001 ;
    Control_C->trivial = 4 ;
    Control_C->worthwhile_dgemm = 512 ;
    Control_C->worthwhile_trsm = 4096 ;
    Control_C->paru_max_threads = 0 ;
    return (PARU_SUCCESS) ;
}

//------------------------------------------------------------------------------
// ParU_C_Analyze: Symbolic analysis is done in this routine. UMFPACK is called
// here and after that some more speciallized symbolic computation is done for
// ParU. ParU_Analyze can be called once and can be used for different
// ParU_C_Factorize calls.
//------------------------------------------------------------------------------

ParU_Info ParU_C_Analyze
(
    // input:
    cholmod_sparse *A,  // input matrix to analyze of size n-by-n
    // output:
    ParU_C_Symbolic **Sym_handle_C,  // output, symbolic analysis
    // control:
    ParU_C_Control *Control_C
)
{
    if (!A || !Sym_handle_C || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_C_Symbolic *Sym_C = PARU_CALLOC (1, ParU_C_Symbolic);
    if (!Sym_C)
    {
        return (PARU_OUT_OF_MEMORY) ;
    }
    ParU_Symbolic *Sym;
    ParU_Info info = ParU_Analyze(A, &Sym, &Control);
    if (info != PARU_SUCCESS)
    {
        PARU_FREE (1, ParU_C_Symbolic, Sym_C);
        return (info) ;
    }
    Sym_C->sym_handle = static_cast<void*>(Sym);
    Sym_C->m = Sym->m;
    Sym_C->n = Sym->n;
    Sym_C->anz = Sym->anz;
    Sym_C->Qfill = Sym->Qfill ;
    Sym_C->paru_strategy = Sym->paru_strategy ;
    Sym_C->umfpack_ordering = Sym->umfpack_ordering ;
    (*Sym_handle_C) = Sym_C;
    return (info) ;
}

//------------------------------------------------------------------------------
// ParU_C_Factorize: Numeric factorization is done in this routine. Scaling and
// making Sx matrix, computing factors and permutations is here.
// ParU_C_Symbolic structure is computed by ParU_C_Analyze and is an input in
// this routine.
//------------------------------------------------------------------------------

ParU_Info ParU_C_Factorize
(
    // input:
    cholmod_sparse *A,          // input matrix to factorize of size n-by-n
    ParU_C_Symbolic *Sym_C,     // symbolic analysis from ParU_Analyze
    // output:
    ParU_C_Numeric **Num_handle_C,    // output numerical factorization
    // control:
    ParU_C_Control *Control_C
)
{
    if (!A || !Sym_C || !Num_handle_C || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_Symbolic *Sym = static_cast<ParU_Symbolic*>(Sym_C->sym_handle);
    ParU_C_Numeric *Num_C = PARU_CALLOC (1, ParU_C_Numeric);
    if (!Num_C)
    {
        return (PARU_OUT_OF_MEMORY) ;
    }

    ParU_Info info;
    ParU_Numeric *Num;
    info = ParU_Factorize(A, Sym, &Num, &Control);
    if (info != PARU_SUCCESS)
    {
        PARU_FREE (1, ParU_C_Numeric, Num_C);
        return info;
    }
    Num_C->num_handle = static_cast<void*>(Num);
    Num_C->rcond = Num->rcond;
    Num_C->Pfin = Num->Pfin ;
    Num_C->Rs = Num->Rs ;
    (*Num_handle_C) = Num_C;
    return (info) ;
}

//------------------------------------------------------------------------------
//--------------------- Solve routines -----------------------------------------
//------------------------------------------------------------------------------

// In all the solve routines Num structure must come with the same Sym struct
// that comes from ParU_Factorize

// x = A\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Axx
(
    // input:
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !x || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_Solve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       x, &Control)) ;
}

// x = L\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Lxx
(
    // input:
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !x || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_LSolve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       x, &Control)) ;
}

// x = U\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Uxx
(
    // input:
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !x || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_USolve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       x, &Control)) ;
}

// x = A\b, for vectors x and b
ParU_Info ParU_C_Solve_Axb
(
    // input:
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    double *b,              // vector of size n-by-1
    // output
    double *x,              // vector of size n-by-1
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !b || !x || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_Solve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       b, x, &Control)) ;
}

// X = A\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_AXX
(
    // input
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !X || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_Solve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       nrhs, X, &Control)) ;
}

// X = L\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_LXX
(
    // input
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !X || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_LSolve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       nrhs, X, &Control)) ;
}

// X = U\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_UXX
(
    // input
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !X || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_USolve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       nrhs, X, &Control)) ;
}

// X = A\B, for matrices X and B
ParU_Info ParU_C_Solve_AXB
(
    // input
    ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_C_Analyze
    ParU_C_Numeric *Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    double *B,              // array of size n-by-nrhs in column-major storage
    // output:
    double *X,              // array of size n-by-nrhs in column-major storage
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Sym_C || !Num_C || !B || !X || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);

    return (ParU_Solve (static_cast<ParU_Symbolic*>(Sym_C->sym_handle),
                       static_cast<ParU_Numeric*>(Num_C->num_handle),
                       nrhs, B, X, &Control)) ;
}

//------------------------------------------------------------------------------
// Perm and InvPerm
//------------------------------------------------------------------------------

// apply permutation to a vector, x=b(p)./s
ParU_Info ParU_C_Perm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control:
    ParU_C_Control *Control_C
)
{
    if (!x || !b || !P || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control ;
    paru_cp_control (&Control, Control_C) ;
    return (ParU_Perm (P, s, b, n, x, &Control)) ;
}

// apply permutation to a matrix, X=B(p,:)./s
ParU_Info ParU_C_Perm_X
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control:
    ParU_C_Control *Control_C
)
{
    if (!X || !B || !P || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control ;
    paru_cp_control (&Control, Control_C) ;
    return (ParU_Perm (P, s, B, nrows, ncols, X, &Control)) ;
}

// apply inverse permutation to a vector, x(p)=b, then scale x=x./s
ParU_Info ParU_C_InvPerm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control
    ParU_C_Control *Control_C
)
{
    if (!x || !b || !P || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control ;
    paru_cp_control (&Control, Control_C) ;
    return (ParU_InvPerm (P, s, b, n, x, &Control)) ;
}

// apply inverse permutation to a matrix, X(p,:)=b, then scale X=X./s
ParU_Info ParU_C_InvPerm_X
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control
    ParU_C_Control *Control_C
)
{
    if (!X || !B || !P || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control ;
    paru_cp_control (&Control, Control_C) ;
    return (ParU_InvPerm (P, s, B, nrows, ncols, X, &Control)) ;
}

//------------------------------------------------------------------------------
//-------------- computing residual --------------------------------------------
//------------------------------------------------------------------------------

// The user provide both x and b
// resid = norm1(b-A*x) / norm1(A)

ParU_Info ParU_C_Residual_bAx
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *x,          // vector of size n
    double *b,          // vector of size n
    // output:
    double *residc,     // residual: norm1(b-A*x) / (norm1(A) * norm1 (x))
    double *anormc,     // 1-norm of A
    double *xnormc,     // 1-norm of x
    // control:
    ParU_C_Control *Control_C
)
{
    if (!A || !x || !b || !residc || !anormc || !xnormc || !Control_C)
    {
        return (PARU_INVALID) ;
    }

    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    double resid, anorm, xnorm;
    ParU_Info info;
    info = ParU_Residual (A, x, b, resid, anorm, xnorm, &Control);
    *residc = resid;
    *anormc = anorm;
    *xnormc = xnorm;
    return info;
}


// resid = norm1(B-A*X) / norm1(A) (multiple rhs)
ParU_Info ParU_C_Residual_BAX
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *X,          // array of size n-by-nrhs
    double *B,          // array of size n-by-nrhs
    int64_t nrhs,
    // output:
    double *residc,     // residual: norm1(B-A*X) / (norm1(A) * norm1 (X))
    double *anormc,     // 1-norm of A
    double *xnormc,     // 1-norm of X
    // control:
    ParU_C_Control *Control_C
)
{
    if (!A || !X || !B || !residc || !anormc || !xnormc || !Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    double resid, anorm, xnorm;
    ParU_Info info;
    info = ParU_Residual (A, X, B, nrhs, resid, anorm, xnorm, &Control);
    *residc = resid;
    *anormc = anorm;
    *xnormc = xnorm;
    return info;
}

//------------------------------------------------------------------------------
//------------ ParU_C_Get_*-----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_Get_INT64
(
    // input:
    const ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric *Num_C,  // numeric factorization from ParU_Factorize
    ParU_Get_enum field,          // field to get
    // output:
    int64_t *result,              // int64_t result: a scalar or an array
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Symbolic *Sym = (Sym_C == NULL) ? NULL :
        static_cast<ParU_Symbolic*>(Sym_C->sym_handle);
    ParU_Numeric *Num = (Num_C == NULL) ? NULL :
        static_cast<ParU_Numeric*>(Num_C->num_handle);
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_Get (Sym, Num, field, result, &Control)) ;
}

ParU_Info ParU_C_Get_FP64
(
    // input:
    const ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric *Num_C,  // numeric factorization from ParU_Factorize
    ParU_Get_enum field,          // field to get
    // output:
    double *result,               // double result: a scalar or an array
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Symbolic *Sym = (Sym_C == NULL) ? NULL : 
        static_cast<ParU_Symbolic*>(Sym_C->sym_handle);
    ParU_Numeric *Num = (Num_C == NULL) ? NULL :
        static_cast<ParU_Numeric*>(Num_C->num_handle);
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_Get (Sym, Num, field, result, &Control)) ;
}

ParU_Info ParU_C_Get_CONSTCHAR
(
    // input:
    const ParU_C_Symbolic *Sym_C, // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric *Num_C,  // numeric factorization from ParU_Factorize
    ParU_Get_enum field,          // field to get
    // output:
    const char **result,          // string result
    // control:
    ParU_C_Control *Control_C
)
{
    if (!Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Symbolic *Sym = (Sym_C == NULL) ? NULL : 
        static_cast<ParU_Symbolic*>(Sym_C->sym_handle);
    ParU_Numeric *Num = (Num_C == NULL) ? NULL :
        static_cast<ParU_Numeric*>(Num_C->num_handle);
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return (ParU_Get (Sym, Num, field, result, &Control)) ;
}

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_FreeNumeric
(
    ParU_C_Numeric **Num_handle_C,    // numeric object to free
    // control:
    ParU_C_Control *Control_C
)
{
    if (Num_handle_C == NULL || *Num_handle_C == NULL)
    {
        // nothing to do
        return (PARU_SUCCESS) ;
    }
    if (!Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_C_Numeric *Num_C = *Num_handle_C;
    ParU_Numeric *Num = static_cast<ParU_Numeric*>(Num_C->num_handle);
    ParU_Info info = ParU_FreeNumeric(&Num, &Control);
    PARU_FREE(1, ParU_C_Numeric, *Num_handle_C);
    return info;
}

ParU_Info ParU_C_FreeSymbolic
(
    ParU_C_Symbolic **Sym_handle_C,   // symbolic object to free
    // control:
    ParU_C_Control *Control_C
)
{
    if (Sym_handle_C == NULL || *Sym_handle_C == NULL)
    {
        // nothing to do
        return (PARU_SUCCESS) ;
    }
    if (!Control_C)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_C_Symbolic *Sym_C = *Sym_handle_C;
    ParU_Symbolic *Sym = static_cast<ParU_Symbolic*>(Sym_C->sym_handle);
    ParU_Info info = ParU_FreeSymbolic(&Sym, &Control);
    PARU_FREE(1, ParU_C_Symbolic, *Sym_handle_C);
    return info;
}

} // extern "C"
