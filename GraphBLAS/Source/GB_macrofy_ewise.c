//------------------------------------------------------------------------------
// GB_macrofy_ewise: construct all macros for GrB_eWise* (Add, Mult, Union)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_ewise           // construct all macros for GrB_eWise
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_BinaryOp binaryop,      // binaryop to macrofy (NULL for GB_wait)
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
)
{

    //--------------------------------------------------------------------------
    // extract the binaryop scode
    //--------------------------------------------------------------------------

    // A and B iso-valued and flipxy (one hex digit)
//  int unused      = GB_RSHIFT (scode, 47, 1) ;
    int A_iso_code  = GB_RSHIFT (scode, 46, 1) ;
    int B_iso_code  = GB_RSHIFT (scode, 45, 1) ;
    bool flipxy     = GB_RSHIFT (scode, 44, 1) ;

    // binary operator (5 hex digits)
    int binop_ecode = GB_RSHIFT (scode, 36, 8) ;
    int zcode       = GB_RSHIFT (scode, 32, 4) ;
    int xcode       = GB_RSHIFT (scode, 28, 4) ;
    int ycode       = GB_RSHIFT (scode, 24, 4) ;

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (scode, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (scode, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (scode, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (scode,  8, 4) ;   // if 0: B is pattern

    bool C_iso = (ccode == 0) ;

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (scode,  6, 2) ;
    int msparsity   = GB_RSHIFT (scode,  4, 2) ;
    int asparsity   = GB_RSHIFT (scode,  2, 2) ;
    int bsparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    GB_macrofy_copyright (fp) ;
    if (binaryop == NULL)
    {  
        // GB_wait: A and B are disjoint and the operator is not applied
        fprintf (fp, "// op: none, type: %s\n\n", ctype->name) ;
    }
    else
    { 
        // normal case
        fprintf (fp, "// op: %s%s, xtype: %s\n\n", binaryop->name,
            flipxy ? " (flipped)" : "", binaryop->xtype->name) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    if (binaryop == NULL)
    { 
        // GB_wait: all types must be the same
        GB_macrofy_types (fp, ctype, NULL, NULL, NULL, NULL, NULL) ;
    }
    else
    { 
        GB_macrofy_types (fp, ctype, atype, btype,
            binaryop->xtype, binaryop->ytype, binaryop->ztype) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for the type names
    //--------------------------------------------------------------------------

    fprintf (fp, "// binary operator types:\n") ;
    if (binaryop == NULL)
    {
        // GB_wait: implicit SECOND operator
        fprintf (fp, "#define GB_X_TYPENAME %s\n", ctype->name) ;
        fprintf (fp, "#define GB_Y_TYPENAME %s\n", ctype->name) ;
        fprintf (fp, "#define GB_Z_TYPENAME %s\n", ctype->name) ;
    }
    else
    {
        fprintf (fp, "#define GB_X_TYPENAME %s\n", binaryop->xtype->name) ;
        fprintf (fp, "#define GB_Y_TYPENAME %s\n", binaryop->ytype->name) ;
        fprintf (fp, "#define GB_Z_TYPENAME %s\n", binaryop->ztype->name) ;
    }

    //--------------------------------------------------------------------------
    // construct macros for the multiply
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// binary operator%s:\n", flipxy ? " (flipped)" : "") ;
    GB_macrofy_binop (fp, "GB_BINOP", flipxy, false, binop_ecode, binaryop) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// C matrix:\n") ;
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

    GB_macrofy_input (fp, "a", "A", (binaryop == NULL) ? ctype :
        (flipxy ? binaryop->ytype : binaryop->xtype),
        atype, asparsity, acode, A_iso_code) ;

    GB_macrofy_input (fp, "b", "B", (binaryop == NULL) ? ctype :
        (flipxy ? binaryop->xtype : binaryop->ytype),
        btype, bsparsity, bcode, B_iso_code) ;

}

