//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_change_factor: change format of a factor object
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// The cholmod_factor object holds a sparse Cholesky factorization in one of
// many formats.  It can be numeric or symbolic, L*L' or L*D*L', simplicial
// or supernodal, packed or unpacked, and its columns can appear in order
// (monotonic) or not (non-monotonic).  Not all combinations are possible.

// The L->xtype field describes the type of the entries:

//      CHOLMOD_PATTERN: L->x and L->z are NULL, no values are stored.
//      CHOLMOD_REAL: L->x is float or double, and L->z is NULL.  The numeric
//          value of the (p)th entry is L->x [p].
//      CHOLMOD_COMPLEX: L->x is float or double, the matrix is complex,
//          where the value of (p)th entry is L->x [2*p] (the real part),
//          and L->x [2*p+1] (the imaginary part).  L->z is NULL.
//      CHOLMOD_ZOMPLEX: L->x and L->z are float or double. The matrix is
//          "zomplex", which means the matrix is mathematically complex, but
//          the real and imaginary parts are held in two separate arrays.
//          The value of the (p)th entry is L->x [p] (the real part), and
//          L->z [p] (the imaginary part).  Supernodal factors are never
//          zomplex.

// This method can change the L->xtype of a factor object L, but only to
// convert it to CHOLMOD_PATTERN (thus freeing all numeric values), or by
// changing a factor of xtype CHOLMOD_PATTERN to any of the three other types
// (thus allocating empty space, or placing a placeholder value of an
// identitity matrix.

// The L->dtype field, just like the CHOLMOD sparse matrix, triplet matrix, and
// dense matrix formats, defines the underlying floating point type: single
// (float) or double, as CHOLMOD_SINGLE or CHOLMOD_DOUBLE.  Matrices of
// different dtypes cannot be mixed.  To convert the xtype and dtype of an
// object, use the cholmod_*_xtype methods (named "_xtype" for backward
// compatibility with CHOLMOD v4 and earlier).  That method will preserve the
// values of the factor L.

// CHOLMOD has the following basic kinds of factor formats:
//
// (1) simplicial symbolic:  this consists of just two arrays of size n.
//      L->Perm contains the fill-reducing ordering, and L->ColCount is an
//      array containing the # of entries in each column of L.  All of the other
//      factor objects contain this information as well.  The simplicial
//      symbolic format does not hold the pattern of the matrix L itself.
//
// (2) simplicial numeric:  this can hold either an LL' or LDL' factorization.
//
//      LDL':  The numeric L matrix itself is unit-diagonal, and the diagonal
//      entries of L are not stored.  The jth column appears in positions
//      L->p [j] to L->p [j] + L->nz [j], in the arrays L->i, L->x, and (if
//      zomplex), L->z.  Thus, L->nz [j] is the # of entries in the jth column,
//      which includes the entry D(j,j) as the first entry held in that column.
//      The columns of L need not appear in order 0, 1, ... n-1 in L->(i,x,z).
//      Instead, a doubly-link list is used (with L->prev and L->next).
//      The value D(j,j) can be negative (that is, the matrix being factorized
//      can be symmetric indefinite but with all principal minor matrices being
//      full rank).
//
//      LL': this is the same format as LL', except the first entry in each
//      column is L(j,j).  The diagonal matrix D is not preset.
//
// (3) supernodal symbolic:  this represents the nonzero pattern of the
//      supernodes for a supernodal factorization, with L->nsuper supernodes.
//      The kth supernode contains columns L->super [k] to L->super [k+1]-1.
//      Its row indices are held in L->s [L->pi [k] ... L->pi [k+1]-1].
//      L->x is not allocated, and is NULL.
//
// (4) supernodal numeric:  This is always an LL' factorization (not LDL').
//      L->x holds then numeric values of each supernode.  The values of
//      the kth supernode (for the real case) are held in
//      L->x [L->px [k] ... L->px [k+1]-1].  The factor can be complex but
//      not zomplex (L->z is never used for a supernodal numeric factor).
//
// Within each column, or which each supernode, the row indices in L->i
// (simplicial) or L->s (supernodal) are kept sorted, from low to high.
//
// This function, cholmod_change_factor and cholmod_l_change_factor, converts
// a factor object between these representations, with some limitations:
//
// (a) a simplicial numeric factor cannot be converted to supernodal.
//
// (b) only a symbolic factor (simplicial or supernodal) can be converted
//      into a supernodal numeric factor.
//
// (c) L->dtype is not changed (single or double precision).
//
// (d) L->xtype can be changed but the numeric contents of L are discarded.
//
// Some of these conversions are only meant for internal use by CHOLMOD itself,
// and they allocate space whose contents are not defined: simplicial
// symbolic to supernodal symbolic, and converting any factor to supernodal
// numeric.  CHOLMOD performs these conversions just before it does its
// numeric factorization.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR(result)             \
{                                           \
    if (Common->status < CHOLMOD_OK)        \
    {                                       \
        return result ;                     \
    }                                       \
}

//------------------------------------------------------------------------------
// grow_column: increase the space for a single column of L
//------------------------------------------------------------------------------

static Int grow_column (Int len, double grow1, double grow2, Int maxlen)
{
    double xlen = (double) len ;
    xlen = grow1 * xlen + grow2 ;
    xlen = MIN (xlen, maxlen) ;
    len = (Int) xlen ;
    len = MAX (1, len) ;
    len = MIN (len, maxlen) ;
    return (len) ;
}

//------------------------------------------------------------------------------
// grow_L: grow the space at the end of L
//------------------------------------------------------------------------------

static Int grow_L (Int lnz, double grow0, Int n)
{
    double xlnz = (double) lnz ;
    xlnz *= grow0 ;
    xlnz = MIN (xlnz, (double) SIZE_MAX) ;
    double d = (double) n ;
    d = (d*d + d) / 2 ;
    xlnz = MIN (xlnz, d) ;
    lnz = (Int) xlnz ;
    return (lnz) ;
}

//------------------------------------------------------------------------------
// t_cholmod_change_factor_*_worker
//------------------------------------------------------------------------------

#define DOUBLE

    #define REAL
    #include "t_cholmod_change_factor_1_worker.c"
    #include "t_cholmod_change_factor_2_worker.c"
    #include "t_cholmod_change_factor_3_worker.c"
    #undef REAL

    #define COMPLEX
    #include "t_cholmod_change_factor_1_worker.c"
    #include "t_cholmod_change_factor_2_worker.c"
    #include "t_cholmod_change_factor_3_worker.c"
    #undef COMPLEX

    #define ZOMPLEX
    #include "t_cholmod_change_factor_1_worker.c"
    #include "t_cholmod_change_factor_2_worker.c"
    #undef ZOMPLEX

#undef  DOUBLE
#define SINGLE

    #define REAL
    #include "t_cholmod_change_factor_1_worker.c"
    #include "t_cholmod_change_factor_2_worker.c"
    #include "t_cholmod_change_factor_3_worker.c"
    #undef REAL

    #define COMPLEX
    #include "t_cholmod_change_factor_1_worker.c"
    #include "t_cholmod_change_factor_2_worker.c"
    #include "t_cholmod_change_factor_3_worker.c"
    #undef COMPLEX

    #define ZOMPLEX
    #include "t_cholmod_change_factor_1_worker.c"
    #include "t_cholmod_change_factor_2_worker.c"
    #undef ZOMPLEX

//------------------------------------------------------------------------------
// natural list: create a doubly-link list of columns, in ordering 0 to n-1
//------------------------------------------------------------------------------

// The head of the link list is always n+1, and the tail is always n, where
// n = L->n.  The actual columns of L are in range 0 to L->n.

static void natural_list (cholmod_factor *L)
{
    // get inputs
    Int *Lnext = (Int *) L->next ;
    Int *Lprev = (Int *) L->prev ;
    ASSERT (Lprev != NULL && Lnext != NULL) ;
    Int n = L->n ;

    // create the head node
    Int head = n+1 ;
    Lnext [head] = 0 ;
    Lprev [head] = EMPTY ;

    // create the tail node
    Int tail = n ;
    Lnext [tail] = EMPTY ;
    Lprev [tail] = n-1 ;

    // link columns 0 to n-1 in increasing order: 0, 1, 2, ... n-1
    for (Int j = 0 ; j < n ; j++)
    {
        Lnext [j] = j+1 ;
        Lprev [j] = j-1 ;
    }

    // the prev node of the first coumn 0 is n+1
    Lprev [0] = head ;

    // the columns appear in order 0, 1, 2, ... n-1 in the link list
    L->is_monotonic = TRUE ;
}


//------------------------------------------------------------------------------
// L_is_packed: return true if no column of L has any extra space after it
//------------------------------------------------------------------------------

// This method is used for debugging only.

#ifndef NDEBUG
static int L_is_packed (cholmod_factor *L, cholmod_common *Common)
{
    Int *Lnz = (Int *) L->nz ;
    Int *Lp  = (Int *) L->p ;
    Int n = L->n ;

    if (Lnz == NULL || Lp == NULL || L->xtype == CHOLMOD_PATTERN || L->is_super)
    {
        // nothing to check: L is intrinsically packed
        return (TRUE) ;
    }

    if (!L->is_monotonic)
    {
        // L is not packed, by definition
        return (FALSE) ;
    }

    for (Int j = 0 ; j < n ; j++)
    {
        PRINT3 (("j: "ID" Lnz "ID" Lp[j+1] "ID" Lp[j] "ID"\n", j, Lnz [j],
                Lp [j+1], Lp [j])) ;
        Int total_space_for_column_j = Lp [j+1] - Lp [j] ;
        Int entries_in_column_j  = Lnz [j] ;
        if (entries_in_column_j != total_space_for_column_j)
        {
            // L(:,j) has extra slack space at the end of the column
            PRINT2 (("L is not packed\n")) ;
            return (FALSE) ;
        }
    }

    // L is packed
    return (TRUE) ;
}
#endif

//------------------------------------------------------------------------------
// alloc_simplicial_num: allocate size-n arrays for simplicial numeric
//------------------------------------------------------------------------------

// Does not allocate L->i, L->x, or L->z, which are larger.
// See also cholmod_alloc_factor, which allocates L->Perm and L->ColCount.

static int alloc_simplicial_num
(
    cholmod_factor *L,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L->xtype == CHOLMOD_PATTERN || L->is_super) ;
    ASSERT (L->p == NULL) ;
    ASSERT (L->nz == NULL) ;
    ASSERT (L->prev == NULL) ;
    ASSERT (L->next == NULL) ;

    size_t n = L->n ;

    //--------------------------------------------------------------------------
    // allocate the four arrays
    //--------------------------------------------------------------------------

    Int *Lp    = CHOLMOD(malloc) (n+1, sizeof (Int), Common) ;
    Int *Lnz   = CHOLMOD(malloc) (n,   sizeof (Int), Common) ;
    Int *Lprev = CHOLMOD(malloc) (n+2, sizeof (Int), Common) ;
    Int *Lnext = CHOLMOD(malloc) (n+2, sizeof (Int), Common) ;

    //--------------------------------------------------------------------------
    // check if out of memory
    //--------------------------------------------------------------------------

    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        CHOLMOD(free) (n+1, sizeof (Int), Lp,    Common) ;
        CHOLMOD(free) (n,   sizeof (Int), Lnz,   Common) ;
        CHOLMOD(free) (n+2, sizeof (Int), Lprev, Common) ;
        CHOLMOD(free) (n+2, sizeof (Int), Lnext, Common) ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // place the arrays in L
    //--------------------------------------------------------------------------

    L->p = Lp ;
    L->nz = Lnz ;
    L->prev = Lprev ;
    L->next = Lnext ;

    //--------------------------------------------------------------------------
    // initialize the link list of the columns of L
    //--------------------------------------------------------------------------

    natural_list (L) ;
    return (TRUE) ;
}

//------------------------------------------------------------------------------
// simplicial_sym_to_super_sym: converts simplicial symbolic to super
//------------------------------------------------------------------------------

// converts a simplicial symbolic factor to supernodal symbolic.  The space
// is allocated but not initialized.

static int simplicial_sym_to_super_sym
(
    cholmod_factor *L,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L->xtype == CHOLMOD_PATTERN && !(L->is_super)) ;

    //--------------------------------------------------------------------------
    // allocate L->super, L->pi, L->px, and L->s
    //--------------------------------------------------------------------------

    Int *Lsuper = CHOLMOD(malloc) (L->nsuper+1, sizeof (Int), Common) ;
    Int *Lpi    = CHOLMOD(malloc) (L->nsuper+1, sizeof (Int), Common) ;
    Int *Lpx    = CHOLMOD(malloc) (L->nsuper+1, sizeof (Int), Common) ;
    Int *Ls     = CHOLMOD(malloc) (L->ssize,    sizeof (Int), Common) ;

    //--------------------------------------------------------------------------
    // check if out of memory
    //--------------------------------------------------------------------------

    if (Common->status < CHOLMOD_OK)
    {
        // out of memory
        CHOLMOD(free) (L->nsuper+1, sizeof (Int), Lsuper, Common) ;
        CHOLMOD(free) (L->nsuper+1, sizeof (Int), Lpi,    Common) ;
        CHOLMOD(free) (L->nsuper+1, sizeof (Int), Lpx,    Common) ;
        CHOLMOD(free) (L->ssize,    sizeof (Int), Ls,     Common) ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // place the arrays in L
    //--------------------------------------------------------------------------

    L->super = Lsuper ;
    L->pi = Lpi ;
    L->px = Lpx ;
    L->s  = Ls ;

    //--------------------------------------------------------------------------
    // revise the header of L (L->dtype is not changed)
    //--------------------------------------------------------------------------

    L->xtype = CHOLMOD_PATTERN ;    // L is symbolic (no L->x, L->i, L->z)
    L->is_super = TRUE ;            // L is now supernodal
    Ls [0] = EMPTY ;                // contents of supernodal pattern undefined
    L->is_ll = TRUE ;               // L is a supernodal LL' factorization
    L->maxcsize = 0 ;               // size of largest update matrix
    L->maxesize = 0 ;               // max rows in any supernode excl tri. part)
    L->minor = L->n ;               // see cholmod.h for a description
    return (TRUE) ;
}

//------------------------------------------------------------------------------
// super_num_to_super_sym: convert numeric supernodal to symbolic
//------------------------------------------------------------------------------

// This method converts a supernodal numeric factor L to supernodal symbolic,
// by freeing the numeric values of all the supernodes. The supernodal
// pattern (L->s) is kept.

static void super_num_to_super_sym
(
    cholmod_factor *L,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L->xtype != CHOLMOD_PATTERN) ;
    ASSERT (L->xtype != CHOLMOD_ZOMPLEX) ;
    ASSERT (L->is_super) ;
    ASSERT (L->is_ll) ;
    DEBUG (CHOLMOD(dump_factor) (L, "supernum to supersym:L input", Common)) ;

    //--------------------------------------------------------------------------
    // free L->x only
    //--------------------------------------------------------------------------

    size_t e = (L->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((L->xtype == CHOLMOD_PATTERN) ? 0 :
                    ((L->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;

    L->x = CHOLMOD(free) (L->xsize, ex, L->x, Common) ;

    //--------------------------------------------------------------------------
    // change the header contents to reflect the supernodal symbolic status
    //--------------------------------------------------------------------------

    L->xtype = CHOLMOD_PATTERN ;    // L is symbolic
    L->minor = L->n ;               // see cholmod.h
    L->is_ll = TRUE ;               // supernodal factor is always LL', not LDL'
    DEBUG (CHOLMOD(dump_factor) (L, "supernum to supersym:L output", Common)) ;
}

//------------------------------------------------------------------------------
// simplicial_sym_to_simplicial_num: convert simplicial numeric to symbolic
//------------------------------------------------------------------------------

// This methods allocates space and converts a simplicial symbolic L to
// simplicial numeric.  L is set to the identity matrix, except in one case
// where the contents of L are not initialized (packed < 0 case).

static void simplicial_sym_to_simplicial_num
(
    cholmod_factor *L,  // factor to modify
    int to_ll,          // if true, convert to LL. if false: to LDL'
    int packed,         // if > 0: L is packed, if 0: L is unpacked,
                        // if < 0: L is packed but contents are not initialized
    int to_xtype,       // the L->xtype (real, complex, or zomplex)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L->xtype == CHOLMOD_PATTERN && !(L->is_super)) ;

    //--------------------------------------------------------------------------
    // allocate space for the simplicial numeric factor (except L->(i,x,z))
    //--------------------------------------------------------------------------

    if (!alloc_simplicial_num (L, Common))
    {
        // out of memory; error status is already in Common->status
        return ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = L->n ;
    Int *Lp = L->p ;
    Int *Lnz = L->nz ;
    Int *ColCount = L->ColCount ;

    //--------------------------------------------------------------------------
    // initialize L->p and L->nz
    //--------------------------------------------------------------------------

    bool ok = true ;
    Int lnz = 0 ;

    if (packed < 0)
    {

        //----------------------------------------------------------------------
        // do not initialize the space
        //----------------------------------------------------------------------

        lnz = L->nzmax ;
        L->nzmax = 0 ;

    }
    else if (packed > 0)
    {

        //----------------------------------------------------------------------
        // initialize the packed LL' or LDL' case (L is identity)
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < n ; j++)
        {
            // ensure ColCount [j] is in the range 1 to n-j
            Int len = ColCount [j] ;
            len = MAX (1, len) ;
            len = MIN (len, n-j) ;
            lnz += len ;
            ok = (lnz >= 0) ;
            if (!ok) break ;
        }
        // each column L(:,j) holds a single diagonal entry
        for (Int j = 0 ; j <= n ; j++)
        {
            Lp [j] = j ;
        }
        for (Int j = 0 ; j < n ; j++)
        {
            Lnz [j] = 1 ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // initialize the unpacked LL' or LDL' case (L is identity)
        //----------------------------------------------------------------------

        // slack space will be added to L below
        double grow0 = Common->grow0 ;
        double grow1 = Common->grow1 ;
        double grow2 = (double) Common->grow2 ;
        grow0 = isnan (grow0) ? 1 : grow0 ;
        grow1 = isnan (grow1) ? 1 : grow1 ;
        Int grow = (grow0 >= 1.0) && (grow1 >= 1.0) && (grow2 > 0) ;

        for (Int j = 0 ; j < n ; j++)
        {

            //------------------------------------------------------------------
            // log the start of L(:,j), containing a single entry
            //------------------------------------------------------------------

            Lp [j] = lnz ;
            Lnz [j] = 1 ;

            //------------------------------------------------------------------
            // ensure ColCount [j] is in the range 1 to n-j
            //------------------------------------------------------------------

            Int len = ColCount [j] ;
            len = MAX (1, len) ;
            len = MIN (len, n-j) ;

            //------------------------------------------------------------------
            // add some slack space to L(:,j)
            //------------------------------------------------------------------

            if (grow)
            {
                len = grow_column (len, grow1, grow2, n-j) ;
            }
            lnz += len ;
            ok = (lnz >= 0) ;
            if (!ok) break ;
        }

        //----------------------------------------------------------------------
        // add slack space at the end of L
        //----------------------------------------------------------------------

        if (ok)
        {
            Lp [n] = lnz ;
            if (grow)
            {
                lnz = grow_L (lnz, grow0, n) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // allocate L->i, L->x, and L->z with the new xtype and existing dtype
    //--------------------------------------------------------------------------

    ASSERT (L->nzmax == 0) ;
    lnz = MAX (1, lnz) ;
    int nint = 1 ;
    if (!ok || !CHOLMOD(realloc_multiple) (lnz, nint, to_xtype + L->dtype,
        &(L->i), NULL, &(L->x), &(L->z), &(L->nzmax), Common))
    {
        // out of memory: convert L back to simplicial symbolic
        CHOLMOD(to_simplicial_sym) (L, to_ll, Common) ;
        return ;
    }

    L->xtype = to_xtype ;
    L->minor = n ;

    //--------------------------------------------------------------------------
    // set L to the identity matrix, if requested
    //--------------------------------------------------------------------------

    if (packed >= 0)
    {

        switch ((L->xtype + L->dtype) % 8)
        {
            case CHOLMOD_SINGLE + CHOLMOD_REAL:
                r_s_cholmod_change_factor_1_worker (L) ;
                break ;

            case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
                c_s_cholmod_change_factor_1_worker (L) ;
                break ;

            case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
                z_s_cholmod_change_factor_1_worker (L) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_REAL:
                r_cholmod_change_factor_1_worker (L) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
                c_cholmod_change_factor_1_worker (L) ;
                break ;

            case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
                z_cholmod_change_factor_1_worker (L) ;
                break ;
        }
    }

    L->is_ll = to_ll ;
}

//------------------------------------------------------------------------------
// change_simplicial_num: change LL' to LDL' or LDL' to LL'
//------------------------------------------------------------------------------

// L must be simplicial numeric.
//
// to_ll:           if true, L is converted to LL'; if false, to LDL'
// to_packed:       if true, L is converted to packed and monotonic
//                  (to_monotonic is treated as true)
//
// to_monotonoic:   if true but to_packed is false, L is converted to monotonic
//                  but the columns are not packed.  Slack space is left in
//                  the columns of L.
//
// If both to_packed and to_monotonic are false:  the columns of L are
//                  converted in place, and neither packed nor made monotonic.
//
// To convert LDL' to LL', columns with D(j,j) <= 0 are left as-is, but L is
// not numerically valid.  The column L(:,j) is unchanged.  If converted back
// to LDL', the column is also left as-is, so that the LDL' can be recovered.
// L->minor is set to the first j where D(j,j) <= 0.

static void change_simplicial_num
(
    cholmod_factor *L,      // factor to modify
    int to_ll,              // if true: convert to LL'; else LDL'
    int to_packed,          // if true: pack the columns of L
    int to_monotonic,       // if true: make columns of L monotonic
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Common != NULL) ;
    DEBUG (CHOLMOD(dump_factor) (L, "change simplnum:L input", Common)) ;
    ASSERT (L->xtype != CHOLMOD_PATTERN) ;
    ASSERT (!L->is_super) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    bool out_of_place = ((to_packed || to_monotonic) && !(L->is_monotonic)) ;
    bool make_ll  = (to_ll && !(L->is_ll)) ;
    bool make_ldl = (!to_ll && L->is_ll) ;

    Int n = L->n ;
    Int *Lp  = (Int *) L->p ;
    Int *Li  = (Int *) L->i ;
    Int *Lnz = (Int *) L->nz ;

    bool grow = false ;
    double grow0 = Common->grow0 ;
    double grow1 = Common->grow1 ;
    double grow2 = (double) Common->grow2 ;
    grow0 = isnan (grow0) ? 1 : grow0 ;
    grow1 = isnan (grow1) ? 1 : grow1 ;

    void *Li2 = NULL ;
    void *Lx2 = NULL ;
    void *Lz2 = NULL ;

    size_t ei = sizeof (Int) ;
    size_t e = (L->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((L->xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;
    size_t ez = e * ((L->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // resize if changing to monotonic and/or packed and not already monotonic
    //--------------------------------------------------------------------------

    Int lnz = 0 ;

    if (out_of_place)
    {

        //----------------------------------------------------------------------
        // out-of-place: construct L in new space (Li2, Lx2, and Lz2)
        //----------------------------------------------------------------------

        // The columns of L are out of order (not monotonic), but L is being
        // changed to being either monotonic, or packed, or both.  Thus, L
        // needs to be resized, in newly allocated space (Li2, Lx2, and Lz2).

        //----------------------------------------------------------------------
        // determine if L should grow
        //----------------------------------------------------------------------

        if (!to_packed)
        {
            grow = (grow0 >= 1.0) && (grow1 >= 1.0) && (grow2 > 0) ;
        }

        //----------------------------------------------------------------------
        // compute the new space for each column of L
        //----------------------------------------------------------------------

        for (Int j = 0 ; j < n ; j++)
        {
            Int len = Lnz [j] ;
            ASSERT (len >= 1 && len <= n-j) ;
            if (grow)
            {
                len = grow_column (len, grow1, grow2, n-j) ;
            }
            ASSERT (len >= Lnz [j] && len <= n-j) ;
            lnz += len ;
            if (lnz <= 0)
            {
                ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
                return ;
            }
        }

        //----------------------------------------------------------------------
        // add additional space at the end of L, if requested
        //----------------------------------------------------------------------

        if (grow)
        {
            lnz = grow_L (lnz, grow0, n) ;
        }

        //----------------------------------------------------------------------
        // allocate Li2, Lx2, and Lz2 (as newly allocated space)
        //----------------------------------------------------------------------

        lnz = MAX (1, lnz) ;
        int nint = 1 ;
        size_t nzmax0 = 0 ;
        CHOLMOD(realloc_multiple) (lnz, nint, L->xtype + L->dtype, &Li2,
            NULL, &Lx2, &Lz2, &nzmax0, Common) ;
        RETURN_IF_ERROR () ;
    }

    //--------------------------------------------------------------------------
    // convert the simplicial L
    //--------------------------------------------------------------------------

    switch ((L->xtype + L->dtype) % 8)
    {
        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_change_factor_2_worker (L, to_packed, Li2, Lx2, Lz2,
                lnz, grow, grow1, grow2, make_ll, out_of_place, make_ldl,
                Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_change_factor_2_worker (L, to_packed, Li2, Lx2, Lz2,
                lnz, grow, grow1, grow2, make_ll, out_of_place, make_ldl,
                Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_change_factor_2_worker (L, to_packed, Li2, Lx2, Lz2,
                lnz, grow, grow1, grow2, make_ll, out_of_place, make_ldl,
                Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_change_factor_2_worker (L, to_packed, Li2, Lx2, Lz2,
                lnz, grow, grow1, grow2, make_ll, out_of_place, make_ldl,
                Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_change_factor_2_worker (L, to_packed, Li2, Lx2, Lz2,
                lnz, grow, grow1, grow2, make_ll, out_of_place, make_ldl,
                Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_change_factor_2_worker (L, to_packed, Li2, Lx2, Lz2,
                lnz, grow, grow1, grow2, make_ll, out_of_place, make_ldl,
                Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // finalize the result L
    //--------------------------------------------------------------------------

    L->is_ll = to_ll ;

    if (out_of_place)
    {

        //----------------------------------------------------------------------
        // free the old space and move the new space into L
        //----------------------------------------------------------------------

        CHOLMOD(free) (L->nzmax, ei, L->i, Common) ;
        CHOLMOD(free) (L->nzmax, ex, L->x, Common) ;
        CHOLMOD(free) (L->nzmax, ez, L->z, Common) ;

        L->i = Li2 ;
        L->x = Lx2 ;
        L->z = Lz2 ;
        L->nzmax = lnz ;

        //----------------------------------------------------------------------
        // revise the link list (columns 0 to n-1 now in natural order)
        //----------------------------------------------------------------------

        natural_list (L) ;
    }

    DEBUG (CHOLMOD(dump_factor) (L, "change simplnum:L output", Common)) ;
}

//------------------------------------------------------------------------------
// super_num_to_simplicial_num: convert supernodal numeric to simplicial numeric
//------------------------------------------------------------------------------

static void super_num_to_simplicial_num
(
    cholmod_factor *L,
    int to_packed,
    int to_ll,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L != NULL) ;
    ASSERT (Common != NULL) ;
    DEBUG (CHOLMOD(dump_factor) (L, "supernum to simplnum:L input", Common)) ;
    ASSERT (L->xtype != CHOLMOD_PATTERN) ;
    ASSERT (L->xtype != CHOLMOD_ZOMPLEX) ;
    ASSERT (L->is_ll) ;
    ASSERT (L->is_super) ;
    ASSERT (L->x != NULL) ;
    ASSERT (L->i == NULL) ;
    ASSERT (L->z == NULL) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = L->n ;
    Int nsuper = L->nsuper ;
    Int *Lpi   = (Int *) L->pi ;
    Int *Lpx   = (Int *) L->px ;
    Int *Ls    = (Int *) L->s ;
    Int *Super = (Int *) L->super ;

    size_t ei = sizeof (Int) ;
    size_t e = (L->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((L->xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;

    //--------------------------------------------------------------------------
    // determine the size of L after conversion to simplicial numeric
    //--------------------------------------------------------------------------

    Int lnz = 0 ;

    if (to_packed)
    {

        //----------------------------------------------------------------------
        // count the # of nonzeros in all supernodes of L
        //----------------------------------------------------------------------

        // Each supernode is lower trapezoidal, with a top part that is lower
        // triangular (with diagonal present) and a bottom part that is
        // rectangular.  In this example below, nscol = 5, so the supernode
        // represents 5 columns of L, and nsrow = 8, which means that the
        // first column of the supernode has 8 entries, including the
        // diagonal.
        //
        //      x . . . .
        //      x x . . .
        //      x x x . .
        //      x x x x .
        //      x x x x x
        //      x x x x x
        //      x x x x x
        //      x x x x x
        //
        // The '.' entries above are in the data structure but not used, and
        // are not copied into the simplicial factor L.  The 'x' entries are
        // used, but some might be exactly equal to zero.  Some of these zeros
        // come from relaxed supernodal algamation, and some come from exact
        // numeric cancellation.  These entries appear in the final
        // simplicial factor L.

        for (Int s = 0 ; s < nsuper ; s++)
        {

            //------------------------------------------------------------------
            // get the supernode
            //------------------------------------------------------------------

            Int k1 = Super [s] ;        // L(:,k1) is 1st col in supernode s
            Int k2 = Super [s+1] ;      // L(:,k2) is 1st col in supernode s+1
            Int psi = Lpi [s] ;         // start of pattern in Ls of supernode s
            Int psend = Lpi [s+1] ;     // start of pattern in Ls of s+1
            Int nsrow = psend - psi ;   // # entries in 1st col of supernode s
            Int nscol = k2 - k1 ;       // # of columns in supernode
            ASSERT (nsrow >= nscol) ;
            Int erows = nsrow - nscol ; // # of rows below triangular part

            //------------------------------------------------------------------
            // count the entries in the supernode, including any exact zeros
            //------------------------------------------------------------------

            // lower triangular part
            lnz += nscol * (nscol+1) / 2 ;

            // rectangular part
            lnz += nscol * erows ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // the supernodal L will not be packed
        //----------------------------------------------------------------------

        // L->x holds all numeric values of the supernodes, and these entries
        // will remain in place.  L->x will not be decreased in size, so Li
        // will have the same size as L->x.

        lnz = L->xsize ;
    }

    ASSERT (lnz >= 0 && lnz <= (Int) (L->xsize)) ;

    //--------------------------------------------------------------------------
    // allocate Li for the pattern L->i of the simplicial factor of L
    //--------------------------------------------------------------------------

    Int *Li = CHOLMOD(malloc) (lnz, ei, Common) ;
    RETURN_IF_ERROR () ;

    //--------------------------------------------------------------------------
    // allocate the size-n arrays for L: L->p, L->nz, L->prev, and L->nex
    //--------------------------------------------------------------------------

    if (!alloc_simplicial_num (L, Common))
    {
        // out of memory
        CHOLMOD(free) (lnz, ei, Li, Common) ;
        return ;
    }

    //--------------------------------------------------------------------------
    // convert the supernodal numeric L into a simplicial numeric L
    //--------------------------------------------------------------------------

    L->i = Li ;
    L->nzmax = lnz ;

    switch ((L->xtype + L->dtype) % 8)
    {
        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_change_factor_3_worker (L, to_packed, to_ll, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_change_factor_3_worker (L, to_packed, to_ll, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_change_factor_3_worker (L, to_packed, to_ll, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_change_factor_3_worker (L, to_packed, to_ll, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // reduce the size of L->x to match L->i (this cannot fail)
    //--------------------------------------------------------------------------

    L->x = CHOLMOD(realloc) (lnz, ex, L->x, &(L->xsize), Common) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // free the supernodal parts of L and return result
    //--------------------------------------------------------------------------

    L->is_super = FALSE ;           // L is now simplicial
    L->is_ll = to_ll ;              // L is LL' or LDL', according to to_ll
    L->pi    = CHOLMOD(free) (nsuper+1, ei, L->pi,    Common) ;
    L->px    = CHOLMOD(free) (nsuper+1, ei, L->px,    Common) ;
    L->super = CHOLMOD(free) (nsuper+1, ei, L->super, Common) ;
    L->s     = CHOLMOD(free) (L->ssize, ei, L->s,     Common) ;
    L->ssize = 0 ;                  // L->s is not present
    L->xsize = 0 ;                  // L->x is not present
    L->nsuper = 0 ;                 // no supernodes
    L->maxesize = 0 ;               // no rows in any supernodes
    L->maxcsize = 0 ;               // largest update matrix is size zero

    DEBUG (CHOLMOD(dump_factor) (L, "supernum to simplnum:L output", Common)) ;
}

//------------------------------------------------------------------------------
// super_sym_to_super_num: convert supernodal symbolic to numeric
//------------------------------------------------------------------------------

static int super_sym_to_super_num
(
    int to_xtype,
    cholmod_factor *L,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (L != NULL) ;
    ASSERT (Common != NULL) ;
    ASSERT (L->xtype == CHOLMOD_PATTERN) ;
    ASSERT (L->is_super) ;
    ASSERT (L->x == NULL) ;
    ASSERT (to_xtype == CHOLMOD_REAL || to_xtype == CHOLMOD_COMPLEX) ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = (L->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((to_xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;
    size_t xs = L->xsize ;

    //--------------------------------------------------------------------------
    // allocate the space
    //--------------------------------------------------------------------------

    double *Lx = CHOLMOD(malloc) (xs, ex, Common) ;
    RETURN_IF_ERROR ((FALSE)) ;

    //--------------------------------------------------------------------------
    // finalize L and return result
    //--------------------------------------------------------------------------

    // clear the first few entries so valgrind is satisfied
    memset (Lx, 0, MIN (2 * sizeof (double), xs*ex)) ;
    L->x = Lx ;             // new space for numeric values
    L->xtype = to_xtype ;   // real or complex
    L->minor = L->n ;
    return (TRUE) ;
}

//==============================================================================
// cholmod_change_factor
//==============================================================================

// Convert a factor L.  Some conversions simply allocate uninitialized space
// that is meant to be filled later.

int CHOLMOD(change_factor)
(
    int to_xtype,       // CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    int to_ll,          // if true: convert to LL'; else to LDL'
    int to_super,       // if true: convert to supernodal; else to simplicial
    int to_packed,      // if true: pack simplicial columns' else: do not pack
    int to_monotonic,   // if true, put simplicial columns in order
    cholmod_factor *L,  // factor to change
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    Common->status = CHOLMOD_OK ;

    to_xtype = to_xtype & 3 ;
    to_ll = to_ll ? 1 : 0 ;
    if (to_super && (to_xtype == CHOLMOD_ZOMPLEX))
    {
        ERROR (CHOLMOD_INVALID, "supernodal zomplex L not supported") ;
        return (FALSE) ;
    }

    PRINT1 (("-----convert from (%d,%d,%d,%d,%d) to (%d,%d,%d,%d,%d)\n",
    L->xtype, L->is_ll, L->is_super, L_is_packed (L, Common), L->is_monotonic,
    to_xtype, to_ll,    to_super,    to_packed,               to_monotonic)) ;

    //--------------------------------------------------------------------------
    // convert the factor L
    //--------------------------------------------------------------------------

    if (to_xtype == CHOLMOD_PATTERN)
    {

        //----------------------------------------------------------------------
        // convert to symbolic
        //----------------------------------------------------------------------

        if (!to_super)
        {

            //------------------------------------------------------------------
            // convert to simplicial symbolic factor (this cannot fail)
            //------------------------------------------------------------------

            CHOLMOD(to_simplicial_sym) (L, to_ll, Common) ;

        }
        else
        {

            //------------------------------------------------------------------
            // convert to supernodal symbolic factor
            //------------------------------------------------------------------

            if (L->xtype != CHOLMOD_PATTERN && L->is_super)
            {
                // convert supernodal numeric to supernodal symbolic,
                // keeping the pattern but freeing the numeric values.
                super_num_to_super_sym (L, Common) ;
            }
            else if (L->xtype == CHOLMOD_PATTERN && !(L->is_super))
            {
                // convert simplicial symbolic to supernodal symbolic,
                // only meant for use internally to CHOLMOD.
                simplicial_sym_to_super_sym (L, Common) ;
            }
            else
            {
                // can't convert simplicial numeric to supernodal symbolic
                ERROR (CHOLMOD_INVALID, "failed to change L") ;
                return (FALSE) ;
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // convert to numeric
        //----------------------------------------------------------------------

        if (to_super)
        {

            //------------------------------------------------------------------
            // convert to supernodal numeric factor
            //------------------------------------------------------------------

            if (L->xtype == CHOLMOD_PATTERN)
            {
                if (L->is_super)
                {
                    // convert supernodal symbolic to supernodal numeric,
                    // only meant for use internally to CHOLMOD.
                    super_sym_to_super_num (to_xtype, L, Common) ;
                }
                else
                {
                    // convert simplicial symbolic to supernodal numeric,
                    // only meant for use internally to CHOLMOD.
                    if (!simplicial_sym_to_super_sym (L, Common))
                    {
                        // failure, convert back to simplicial symbolic
                        CHOLMOD(to_simplicial_sym) (L, to_ll, Common) ;
                        return (FALSE) ;
                    }
                    // convert supernodal symbolic to supernodal numeric
                    super_sym_to_super_num (to_xtype, L, Common) ;
                }
            }
            else
            {
                // nothing to do if already supernodal numeric
                if (!(L->is_super))
                {
                    ERROR (CHOLMOD_INVALID, "failed to change L") ;
                    return (FALSE) ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // convert any factor to simplicial numeric
            //------------------------------------------------------------------

            if (L->xtype == CHOLMOD_PATTERN && !(L->is_super))
            {
                // convert simplicial symbolic to simplicial numeric (L=D=I)
                simplicial_sym_to_simplicial_num (L, to_ll, to_packed,
                        to_xtype, Common) ;
            }
            else if (L->xtype != CHOLMOD_PATTERN && L->is_super)
            {
                // convert a supernodal LL' to simplicial numeric
                super_num_to_simplicial_num (L, to_packed, to_ll, Common) ;
            }
            else if (L->xtype == CHOLMOD_PATTERN && L->is_super)
            {
                // convert a supernodal symbolic to simplicial numeric (L=D=I)
                CHOLMOD(to_simplicial_sym) (L, to_ll, Common) ;
                simplicial_sym_to_simplicial_num (L, to_ll, to_packed,
                    to_xtype, Common) ;
            }
            else
            {
                // change a simplicial numeric factor: change LL' to LDL', LDL'
                // to LL', or leave as-is.  pack the columns of L, or leave
                // as-is.  Ensure the columns are monotonic, or leave as-is.
                change_simplicial_num (L, to_ll, to_packed, to_monotonic,
                    Common) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (Common->status >= CHOLMOD_OK) ;
}

