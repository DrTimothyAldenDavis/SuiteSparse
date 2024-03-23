//------------------------------------------------------------------------------
// GB_macrofy_ewise: construct all macros for ewise methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
    GrB_BinaryOp binaryop,      // binaryop to macrofy
    GrB_Type ctype,
    GrB_Type atype,             // NULL for apply bind1st
    GrB_Type btype              // NULL for apply bind2nd
)
{

    //--------------------------------------------------------------------------
    // extract the binaryop scode
    //--------------------------------------------------------------------------

    // method (3 bits)
//  bool is_emult   = GB_RSHIFT (scode, 50, 1) ;
//  bool is_union   = GB_RSHIFT (scode, 49, 1) ;
    bool copy_to_C  = GB_RSHIFT (scode, 48, 1) ;

    // C in, A, and B iso-valued and flipxy (one hex digit)
    bool C_in_iso   = GB_RSHIFT (scode, 47, 1) ;
    int A_iso_code  = GB_RSHIFT (scode, 46, 1) ;
    int B_iso_code  = GB_RSHIFT (scode, 45, 1) ;
    bool flipxy     = GB_RSHIFT (scode, 44, 1) ;

    // binary operator (5 hex digits)
    int binop_ecode = GB_RSHIFT (scode, 36, 8) ;
//  int zcode       = GB_RSHIFT (scode, 32, 4) ;
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

    GrB_Type xtype, ytype, ztype ;
    const char *xtype_name, *ytype_name, *ztype_name ;
    ASSERT_BINARYOP_OK (binaryop, "binaryop to macrofy", GB0) ;

    if (C_iso)
    { 
        // values of C are not computed by the kernel
        xtype_name = "GB_void" ;
        ytype_name = "GB_void" ;
        ztype_name = "GB_void" ;
        xtype = NULL ;
        ytype = NULL ;
        ztype = NULL ;
        fprintf (fp, "// op: symbolic only (C is iso)\n\n") ;
    }
    else
    { 
        // general case
        xtype = binaryop->xtype ;
        ytype = binaryop->ytype ;
        ztype = binaryop->ztype ;
        xtype_name = xtype->name ;
        ytype_name = ytype->name ;
        ztype_name = ztype->name ;
        if (binaryop->hash == 0)
        { 
            // builtin operator
            fprintf (fp, "// op: (%s%s, %s)\n\n",
                binaryop->name, flipxy ? " (flipped)" : "", xtype_name) ;
        }
        else
        { 
            // user-defined operator, or created by GB_wait
            fprintf (fp,
                "// op: %s%s%s, ztype: %s, xtype: %s, ytype: %s\n\n",
                (binaryop->opcode == GB_SECOND_binop_code) ? "2nd_" : "",
                binaryop->name, flipxy ? " (flipped)" : "",
                ztype_name, xtype_name, ytype_name) ;
        }
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    if (!C_iso)
    { 
        GB_macrofy_typedefs (fp, ctype,
            (acode == 0 || acode == 15) ? NULL : atype,
            (bcode == 0 || bcode == 15) ? NULL : btype,
            xtype, ytype, ztype) ;
    }

    fprintf (fp, "// binary operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the binary operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// binary operator%s:\n", flipxy ? " (flipped)" : "") ;
    GB_macrofy_binop (fp, "GB_BINOP", flipxy, false, true, binop_ecode, C_iso,
        binaryop, NULL, NULL, NULL) ;

    if (binaryop->opcode == GB_SECOND_binop_code)
    { 
        fprintf (fp, "#define GB_OP_IS_SECOND 1\n") ;
    }

    GB_macrofy_cast_copy (fp, "C", "A", (C_iso || !copy_to_C) ? NULL : ctype,
            (acode == 0 || acode == 15) ? NULL : atype, A_iso_code) ;

    GB_macrofy_cast_copy (fp, "C", "B", (C_iso || !copy_to_C) ? NULL : ctype,
            (bcode == 0 || bcode == 15) ? NULL : btype, B_iso_code) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, C_iso,
        C_in_iso) ;

    fprintf (fp, "#define GB_EWISEOP(Cx,p,aij,bij,i,j)") ;
    if (C_iso)
    { 
        fprintf (fp, "\n") ;
    }
    else if (ctype == ztype)
    { 
        fprintf (fp, " GB_BINOP (Cx [p], aij, bij, i, j)\n") ;
    }
    else
    { 
        fprintf (fp, " \\\n"
            "{                                      \\\n"
            "    GB_Z_TYPE z ;                      \\\n"
            "    GB_BINOP (z, aij, bij, i, j) ;     \\\n"
            "    GB_PUTC (z, Cx, p) ;               \\\n"
            "}\n") ;
    }

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity) ;

    //--------------------------------------------------------------------------
    // construct the macros for A and B
    //--------------------------------------------------------------------------

    // These methods create macros for defining the types of A and B, as well
    // as accessing the entries to provide inputs to the operator.  A and B
    // maybe be valued but not used for the operator.  For example, eWiseAdd
    // with the PAIR operator defines GB_DECLAREA, GB_GETA GB_DECLAREB, and
    // GB_GETB as empty, because the values of A and B are not needed for the
    // operator.  However, acode and bcode will not be 0, and GB_A_TYPE and
    // GB_B_TYPE will be defined, because the entries from A and B can bypass
    // the operator and be directly copied into C.

    // if flipxy false:  A is typecasted to x, and B is typecasted to y.
    // if flipxy true:   A is typecasted to y, and B is typecasted to x.

    if (xcode == 0)
    { 
        xtype = NULL ;
    }
    if (ycode == 0)
    { 
        ytype = NULL ;
    }

    GB_macrofy_input (fp, "a", "A", "A", true, flipxy ? ytype : xtype,
        atype, asparsity, acode, A_iso_code, -1) ;

    GB_macrofy_input (fp, "b", "B", "B", true, flipxy ? xtype : ytype,
        btype, bsparsity, bcode, B_iso_code, -1) ;

    //--------------------------------------------------------------------------
    // include the final default definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"GB_ewise_shared_definitions.h\"\n") ;
}

