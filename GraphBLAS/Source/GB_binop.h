//------------------------------------------------------------------------------
// GB_binop.h: definitions for binary operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_BINOP_H
#define GB_BINOP_H

bool GB_binop_builtin               // true if binary operator is builtin
(
    // inputs:
    const GrB_Type A_type,
    const bool A_is_pattern,        // true if only the pattern of A is used
    const GrB_Type B_type,
    const bool B_is_pattern,        // true if only the pattern of B is used
    const GrB_BinaryOp op,          // binary operator; may be NULL
    const bool flipxy,              // true if z=op(y,x), flipping x and y
    // outputs, unused by caller if this function returns false
    GB_Opcode *opcode,              // opcode for the binary operator
    GB_Type_code *xcode,            // type code for x input
    GB_Type_code *ycode,            // type code for y input
    GB_Type_code *zcode             // type code for z output
) ;

GB_Opcode GB_binop_flip     // flipped opcode
(
    GB_Opcode opcode        // opcode to flip
) ;

GB_PUBLIC   // accessed by the MATLAB interface only
GB_Opcode GB_boolean_rename     // renamed opcode
(
    const GB_Opcode opcode      // opcode to rename
) ;

#endif

