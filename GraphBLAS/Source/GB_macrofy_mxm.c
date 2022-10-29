//------------------------------------------------------------------------------
// GB_macrofy_mxm: construct all macros for a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_mxm        // construct all macros for GrB_mxm
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_Semiring semiring,  // the semiring to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
)
{

    //--------------------------------------------------------------------------
    // extract the semiring scode
    //--------------------------------------------------------------------------

    // monoid (4 hex digits)
//  int unused      = GB_RSHIFT (scode, 63, 1) ;
    int add_ecode   = GB_RSHIFT (scode, 58, 5) ;
    int id_ecode    = GB_RSHIFT (scode, 53, 5) ;
    int term_ecode  = GB_RSHIFT (scode, 48, 5) ;
    bool is_term    = (term_ecode < 30) ;

    // A and B iso-valued and flipxy (one hex digit)
//  int unused      = GB_RSHIFT (scode, 47, 1) ;
    int A_iso_code  = GB_RSHIFT (scode, 46, 1) ;
    int B_iso_code  = GB_RSHIFT (scode, 45, 1) ;
    bool flipxy     = GB_RSHIFT (scode, 44, 1) ;

    // multiplier (5 hex digits)
    int mult_ecode  = GB_RSHIFT (scode, 36, 8) ;
    int zcode       = GB_RSHIFT (scode, 32, 4) ;
    int xcode       = GB_RSHIFT (scode, 28, 4) ;
    int ycode       = GB_RSHIFT (scode, 24, 4) ;

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (scode, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (scode, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (scode, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (scode,  8, 4) ;   // if 0: B is pattern

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (scode,  6, 2) ;
    int msparsity   = GB_RSHIFT (scode,  4, 2) ;
    int asparsity   = GB_RSHIFT (scode,  2, 2) ;
    int bsparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // construct the semiring name
    //--------------------------------------------------------------------------

    GrB_Monoid add = semiring->add ;
    GrB_BinaryOp mult = semiring->multiply ;
    GrB_BinaryOp addop = add->op ;

    GB_macrofy_copyright (fp) ;
    fprintf (fp, "// semiring: (%s, %s%s, %s)\n\n",
        addop->name, mult->name, flipxy ? " (flipped)" : "",
        mult->xtype->name) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_types (fp, ctype, atype, btype,
        mult->xtype, mult->ytype, mult->ztype) ;

    //--------------------------------------------------------------------------
    // construct the macros for the type names
    //--------------------------------------------------------------------------

    fprintf (fp, "// semiring types:\n") ;
    fprintf (fp, "#define GB_X_TYPENAME %s\n", mult->xtype->name) ;
    fprintf (fp, "#define GB_Y_TYPENAME %s\n", mult->ytype->name) ;
    fprintf (fp, "#define GB_Z_TYPENAME %s\n", mult->ztype->name) ;

    //--------------------------------------------------------------------------
    // construct the monoid macros
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// additive monoid:\n") ;
    GB_macrofy_monoid (fp, add_ecode, id_ecode, term_ecode, add) ;

    //--------------------------------------------------------------------------
    // construct macros for the multiply
    //--------------------------------------------------------------------------

    // do not print the user-defined multiplicative function if it is identical
    // to the user-defined additive function.
    fprintf (fp, "\n// multiplicative operator%s:\n",
        flipxy ? " (flipped)" : "") ;
    GB_macrofy_binop (fp, "GB_MULT", flipxy, false, mult_ecode, mult) ;

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// special cases:\n") ;

    // semiring is plus_pair_real
    bool is_plus_pair_real =
        (add_ecode == 11 // plus monoid
        && mult_ecode == 133 // pair multiplicative operator
        && !(zcode == GB_FC32_code || zcode == GB_FC64_code)) ; // real

    fprintf (fp, "#define GB_IS_PLUS_PAIR_REAL_SEMIRING %d\n",
        is_plus_pair_real) ;

    // can ignore overflow in ztype when accumulating the result via the monoid
    bool ztype_ignore_overflow = (
        zcode == GB_INT64_code || zcode == GB_UINT64_code ||
        zcode == GB_FP32_code  || zcode == GB_FP64_code ||
        zcode == GB_FC32_code  || zcode == GB_FC64_code) ;

    // note "CTYPE" is in the name in the CPU kernels (fix them to use ZTYPE)
    fprintf (fp, "#define GB_ZTYPE_IGNORE_OVERFLOW %d\n\n",
        ztype_ignore_overflow) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    fprintf (fp, "// C matrix:\n") ;
    bool C_iso = (ccode == 0) ;
    if (C_iso)
    {
        fprintf (fp, "#define GB_PUTC(blob)\n") ;
        fprintf (fp, "#define GB_C_ISO 1\n") ;
    }
    else
    {
        fprintf (fp, "#define GB_PUTC(blob) blob\n") ;
        fprintf (fp, "#define GB_C_ISO 0\n") ;
    }
    GB_macrofy_sparsity (fp, "C", csparsity) ;
    fprintf (fp, "#define GB_C_TYPENAME %s\n\n", ctype->name) ;

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode) ;
    GB_macrofy_sparsity (fp, "M", msparsity) ;

    //--------------------------------------------------------------------------
    // construct the macros for A and B
    //--------------------------------------------------------------------------

    // if flipxy false:  A is typecasted to x, and B is typecasted to y.
    // if flipxy true:   A is typecasted to y, and B is typecasted to x.

    GB_macrofy_input (fp, "a", "A", flipxy ? mult->ytype : mult->xtype,
        atype, asparsity, acode, A_iso_code) ;

    GB_macrofy_input (fp, "b", "B", flipxy ? mult->xtype : mult->ytype,
        btype, bsparsity, bcode, B_iso_code) ;

}

