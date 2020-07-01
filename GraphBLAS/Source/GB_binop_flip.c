//------------------------------------------------------------------------------
// GB_binop_flip:  flip a binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_binop.h"

GB_Opcode GB_binop_flip     // flipped opcode
(
    GB_Opcode opcode        // opcode to flip
)
{

    switch (opcode)
    {
        // swap FIRST and SECOND
        case GB_FIRST_opcode  : return (GB_SECOND_opcode) ;
        case GB_SECOND_opcode : return (GB_FIRST_opcode) ;

        // swap LT and GT
        case GB_GT_opcode     : return (GB_LT_opcode) ;
        case GB_LT_opcode     : return (GB_GT_opcode) ;

        // swap LE and GE
        case GB_GE_opcode     : return (GB_LE_opcode) ;
        case GB_LE_opcode     : return (GB_GE_opcode) ;

        // swap ISLT and ISGT
        case GB_ISGT_opcode   : return (GB_ISLT_opcode) ;
        case GB_ISLT_opcode   : return (GB_ISGT_opcode) ;

        // swap ISLE and ISGE
        case GB_ISGE_opcode   : return (GB_ISLE_opcode) ;
        case GB_ISLE_opcode   : return (GB_ISGE_opcode) ;

        // swap DIV and RDIV
        case GB_DIV_opcode    : return (GB_RDIV_opcode) ;
        case GB_RDIV_opcode   : return (GB_DIV_opcode) ;

        // swap MINUS and RMINUS
        case GB_MINUS_opcode  : return (GB_RMINUS_opcode) ;
        case GB_RMINUS_opcode : return (GB_MINUS_opcode) ;

        // these operators do not have flipped versions:
        // POW, BGET, BSET, BCLR, BSHIFT, ATAN2, FMOD, REMAINDER, COPYSIGN,
        // LDEXP, CMPLX, and user-defined operators.

        // these operators are commutative; they are their own flipped ops:
        // PLUS, TIMES, PAIR, ANY, ISEQ, ISNE, EQ, NE, MIN, MAX, LOR, LAND,
        // LXOR, LXNOR, HYPOT, BOR, BAND, BXOR, BXNOR.
        default :
            return (opcode) ;
    }
}

