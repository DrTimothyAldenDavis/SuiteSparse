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
// ParU_C_InitControl: allocate Control and set to default values
//------------------------------------------------------------------------------

ParU_Info ParU_C_InitControl (ParU_C_Control *Control_C_handle)
{
    if (Control_C_handle == NULL)
    {
        // null pointer on input
        return (PARU_INVALID) ;
    }
    ParU_C_Control Control_C = PARU_CALLOC (1, ParU_C_Control_struct) ;
    if (Control_C == NULL)
    {
        // out of memory
        return (PARU_OUT_OF_MEMORY) ;
    }
    ParU_Control Control = NULL ;
    ParU_Info info = ParU_InitControl (&Control) ;
    if (info != PARU_SUCCESS)
    {
        // out of memory
        PARU_FREE (1, ParU_C_Control_struct, Control_C) ;
        return (info) ;
    }
    Control_C->control_handle = static_cast<void*>(Control) ;
    (*Control_C_handle) = Control_C ;
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
    ParU_C_Symbolic *Sym_handle_C,  // output, symbolic analysis
    // control:
    ParU_C_Control Control_C
)
{
    if (!A || !Sym_handle_C)
    {
        return (PARU_INVALID) ;
    }

    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;


    ParU_C_Symbolic Sym_C = PARU_CALLOC (1, ParU_C_Symbolic_struct);
    if (!Sym_C)
    {
        return (PARU_OUT_OF_MEMORY) ;
    }
    ParU_Symbolic Sym ;
    ParU_Info info = ParU_Analyze(A, &Sym, Control);
    if (info != PARU_SUCCESS)
    {
        PARU_FREE (1, ParU_C_Symbolic_struct, Sym_C);
        return (info) ;
    }
    Sym_C->sym_handle = static_cast<void*>(Sym);
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
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    // output:
    ParU_C_Numeric *Num_handle_C,    // output numerical factorization
    // control:
    ParU_C_Control Control_C
)
{
    if (!A || !Sym_C || !Num_handle_C)
    {
        return (PARU_INVALID) ;
    }

    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;

    ParU_Symbolic Sym = static_cast<ParU_Symbolic>(Sym_C->sym_handle);
    ParU_C_Numeric Num_C = PARU_CALLOC (1, ParU_C_Numeric_struct) ;
    if (!Num_C)
    {
        return (PARU_OUT_OF_MEMORY) ;
    }

    ParU_Info info;
    ParU_Numeric Num ;
    info = ParU_Factorize(A, Sym, &Num, Control);
    if (info != PARU_SUCCESS)
    {
        PARU_FREE (1, ParU_C_Numeric_struct, Num_C);
        return info;
    }
    Num_C->num_handle = static_cast<void*>(Num);
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
    const ParU_C_Symbolic Sym_C, // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !x)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Solve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                        static_cast<ParU_Numeric>(Num_C->num_handle),
                        x, Control)) ;
}

// x = L\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Lxx
(
    // input:
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !x)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_LSolve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                         static_cast<ParU_Numeric>(Num_C->num_handle),
                         x, Control)) ;
}

// x = U\x, where right-hand side is overwritten with the solution x.
ParU_Info ParU_C_Solve_Uxx
(
    // input:
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !x)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_USolve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                         static_cast<ParU_Numeric>(Num_C->num_handle),
                         x, Control)) ;
}

// x = A\b, for vectors x and b
ParU_Info ParU_C_Solve_Axb
(
    // input:
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    double *b,              // vector of size n-by-1
    // output
    double *x,              // vector of size n-by-1
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !b || !x)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Solve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                        static_cast<ParU_Numeric>(Num_C->num_handle),
                        b, x, Control)) ;
}

// X = A\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_AXX
(
    // input
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !X)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Solve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                        static_cast<ParU_Numeric>(Num_C->num_handle),
                        nrhs, X, Control)) ;
}

// X = L\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_LXX
(
    // input
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !X)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_LSolve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                         static_cast<ParU_Numeric>(Num_C->num_handle),
                         nrhs, X, Control)) ;
}

// X = U\X, where right-hand side is overwritten with the solution X.
ParU_Info ParU_C_Solve_UXX
(
    // input
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    // input/output:
    double *X,              // array of size n-by-nrhs in column-major storage,
                            // right-hand-side on input, solution on output.
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !X)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_USolve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                         static_cast<ParU_Numeric>(Num_C->num_handle),
                         nrhs, X, Control)) ;
}

// X = A\B, for matrices X and B
ParU_Info ParU_C_Solve_AXB
(
    // input
    const ParU_C_Symbolic Sym_C,    // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,  // numeric factorization from ParU_C_Factorize
    int64_t nrhs,
    double *B,              // array of size n-by-nrhs in column-major storage
    // output:
    double *X,              // array of size n-by-nrhs in column-major storage
    // control:
    ParU_C_Control Control_C
)
{
    if (!Sym_C || !Num_C || !B || !X)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Solve (static_cast<ParU_Symbolic>(Sym_C->sym_handle),
                        static_cast<ParU_Numeric>(Num_C->num_handle),
                         nrhs, B, X, Control)) ;
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
    ParU_C_Control Control_C
)
{
    if (!x || !b || !P)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Perm (P, s, b, n, x, Control)) ;
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
    ParU_C_Control Control_C
)
{
    if (!X || !B || !P)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Perm (P, s, B, nrows, ncols, X, Control)) ;
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
    ParU_C_Control Control_C
)
{
    if (!x || !b || !P)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_InvPerm (P, s, b, n, x, Control)) ;
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
    ParU_C_Control Control_C
)
{
    if (!X || !B || !P)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_InvPerm (P, s, B, nrows, ncols, X, Control)) ;
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
    ParU_C_Control Control_C
)
{
    if (!A || !x || !b || !residc || !anormc || !xnormc)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    double resid, anorm, xnorm;
    ParU_Info info;
    info = ParU_Residual (A, x, b, resid, anorm, xnorm, Control);
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
    ParU_C_Control Control_C
)
{
    if (!A || !X || !B || !residc || !anormc || !xnormc)
    {
        return (PARU_INVALID) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    double resid, anorm, xnorm;
    ParU_Info info;
    info = ParU_Residual (A, X, B, nrhs, resid, anorm, xnorm, Control);
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
    const ParU_C_Symbolic Sym_C,  // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,   // numeric factorization from ParU_C_Factorize
    ParU_Get_enum field,          // field to get
    // output:
    int64_t *result,              // int64_t result: a scalar or an array
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Symbolic Sym = (Sym_C == NULL) ? NULL :
        static_cast<ParU_Symbolic>(Sym_C->sym_handle);
    ParU_Numeric Num = (Num_C == NULL) ? NULL :
        static_cast<ParU_Numeric>(Num_C->num_handle);
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Get (Sym, Num, field, result, Control)) ;
}

ParU_Info ParU_C_Get_FP64
(
    // input:
    const ParU_C_Symbolic Sym_C,  // symbolic analysis from ParU_Analyze
    const ParU_C_Numeric Num_C,   // numeric factorization from ParU_C_Factorize
    ParU_Get_enum field,          // field to get
    // output:
    double *result,               // double result: a scalar or an array
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Symbolic Sym = (Sym_C == NULL) ? NULL :
        static_cast<ParU_Symbolic>(Sym_C->sym_handle);
    ParU_Numeric Num = (Num_C == NULL) ? NULL :
        static_cast<ParU_Numeric>(Num_C->num_handle);
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Get (Sym, Num, field, result, Control)) ;
}

ParU_Info ParU_C_Get_Control_CONSTCHAR
(
    // input:
    ParU_Control_enum field,      // field to get
    // output:
    const char **result,          // string result
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Get (field, result, Control)) ;
}

ParU_Info ParU_C_Get_Control_INT64
(
    // input:
    ParU_Control_enum field,      // field to get
    // output:
    int64_t *result,              // int64_t result: a scalar or an array
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Get (field, result, Control)) ;
}

ParU_Info ParU_C_Get_Control_FP64
(
    // input:
    ParU_Control_enum field,      // field to get
    // output:
    double *result,               // int64_t result: a scalar or an array
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Get (field, result, Control)) ;
}

//------------------------------------------------------------------------------
//------------ ParU_C_Set_Control_* --------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_Set_Control_INT64      // set int64_t parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    int64_t c,                  // value to set it to
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Set (field, c, Control)) ;
}

ParU_Info ParU_C_Set_Control_FP64       // set double parameter in Control
(
    // input
    ParU_Control_enum field,    // field to set
    double c,                   // value to set it to
    // control:
    ParU_C_Control Control_C
)
{
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    return (ParU_Set (field, c, Control)) ;
}

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Info ParU_C_FreeNumeric
(
    ParU_C_Numeric *Num_handle_C,    // numeric object to free
    // control:
    ParU_C_Control Control_C
)
{
    if (Num_handle_C == NULL || *Num_handle_C == NULL)
    {
        // nothing to do
        return (PARU_SUCCESS) ;
    }
    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    ParU_C_Numeric Num_C = (*Num_handle_C) ;
    ParU_Numeric Num = static_cast<ParU_Numeric>(Num_C->num_handle);
    ParU_Info info = ParU_FreeNumeric(&Num, Control) ;
    PARU_FREE(1, ParU_C_Numeric_struct, *Num_handle_C) ;
    return info;
}

ParU_Info ParU_C_FreeSymbolic
(
    ParU_C_Symbolic *Sym_handle_C,   // symbolic object to free
    // control:
    ParU_C_Control Control_C
)
{
    if (Sym_handle_C == NULL || *Sym_handle_C == NULL)
    {
        // nothing to do
        return (PARU_SUCCESS) ;
    }

    ParU_Control Control = (Control_C == NULL) ? NULL :
        static_cast<ParU_Control>(Control_C->control_handle) ;
    ParU_C_Symbolic Sym_C = (*Sym_handle_C) ;
    ParU_Symbolic Sym = static_cast<ParU_Symbolic>(Sym_C->sym_handle);
    ParU_Info info = ParU_FreeSymbolic(&Sym, Control);
    PARU_FREE(1, ParU_C_Symbolic_struct, *Sym_handle_C);
    return info;
}

ParU_Info ParU_C_FreeControl
(
    ParU_C_Control *Control_handle_C    // control object to free
)
{
    if (Control_handle_C == NULL || *Control_handle_C == NULL)
    {
        // nothing to do
        return (PARU_SUCCESS) ;
    }
    ParU_C_Control Control_C = (*Control_handle_C) ;
    ParU_Control Control = static_cast<ParU_Control>(Control_C->control_handle) ;
    ParU_Info info = ParU_FreeControl (&Control) ;
    PARU_FREE (1, ParU_C_Control_struct, *Control_handle_C) ;
    return info;
}

} // extern "C"
