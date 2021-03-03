//SPDX-License-Identifier: Apache-2.0

#include "GB.h"
#include "GB_cuda_stringify.h"

// The binop macro generates an expression, not a full statement (there
// is no semicolon).

// for example:
// #define GB_MULT(x,y) ((x) * (y))

void GB_cuda_stringify_binop
(
    // output:
    char *code_string,  // string with the #define macro
    // input:
    const char *macro_name,   // name of macro to construct
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code zcode  // op->ztype->code of the operator
)
{
    const char *op_string ;
    int ecode ;
    GB_cuda_enumify_binop (&ecode, opcode, zcode) ;
    GB_cuda_charify_binop (&op_string, ecode, for_semiring) ;
    GB_cuda_macrofy_binop (code_string, macro_name, op_string) ;
}

void GB_cuda_enumify_binop
(
    // output:
    int *ecode,         // enumerated operator, in range 0 to ... (-1 on failure)
    // input:
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code zcode  // op->ztype->code of the operator
//  bool for_semiring   // true for A*B, false for A+B or A.*B (not needed)
)
{
    int e = -1 ;

    switch (opcode)
    {

        case GB_FIRST_opcode :    // z = x

            e = 0 ; // "(x)" ;
            break ;

        case GB_ANY_opcode :
        case GB_SECOND_opcode :   // z = y

            e = 1 ; // "(y)" ;
            break ;

        case GB_MIN_opcode :      //  z = min(x,y)

            switch (zcode)
            {
                case GB_FP32_code    : e = 2 ; break ;  // "fminf (x,y)" ;
                case GB_FP64_code    : e = 3 ; break ;  // "fmin (x,y)" ;
                default              : e = 4 ; break ;  // "GB_IMIN (x,y)" ;
            }
            break ;

        case GB_MAX_opcode :      // z = max(x,y)

            switch (zcode)
            {
                case GB_FP32_code    : e = 5 ; break ;  // "fmaxf (x,y)" ;
                case GB_FP64_code    : e = 6 ; break ;  // "fmax (x,y)" ;
                default              : e = 7 ; break ;  // "GB_IMAX (x,y)" ;
            }
            break ;

        case GB_PLUS_opcode :     // z = x + y

            e = 8 ; break ;  // "(x) + (y)" ;

        case GB_MINUS_opcode :    // z = x - y

            e = 9 ; break ;  // "(x) - (y)" ;

        case GB_RMINUS_opcode :   // z = y - x

            e = 10 ; break ;  // "(y) - (x)" ;

        case GB_TIMES_opcode :    // z = x * y

            e = 11 ; break ;  // "(x) * (y)" ;

        case GB_DIV_opcode :      // z = x / y ;

            switch (zcode)
            {
                case GB_INT8_code   : e = 12 ; break ; // "GB_IDIV_SIGNED(x,y,8)"
                case GB_INT16_code  : e = 13 ; break ; // "GB_IDIV_SIGNED(x,y,16)"
                case GB_INT32_code  : e = 14 ; break ; // "GB_IDIV_SIGNED(x,y,32)"
                case GB_INT64_code  : e = 15 ; break ; // "GB_IDIV_SIGNED(x,y,64)"
                case GB_UINT8_code  : e = 16 ; break ; // "GB_IDIV_UNSIGNED(x,y,8)"
                case GB_UINT16_code : e = 17 ; break ; // "GB_IDIV_UNSIGNED(x,y,16)"
                case GB_UINT32_code : e = 18 ; break ; // "GB_IDIV_UNSIGNED(x,y,32)"
                case GB_UINT64_code : e = 19 ; break ; // "GB_IDIV_UNSIGNED(x,y,64)"
                default             : e = 20 ; break ; // "(x) / (y)"
            }
            break ;

        case GB_RDIV_opcode :      // z = y / x ;

            switch (zcode)
            {
                case GB_INT8_code   : e = 21 ; break ; // GB_IDIV_SIGNED(y,x,8)
                case GB_INT16_code  : e = 22 ; break ; // GB_IDIV_SIGNED(y,x,16)"
                case GB_INT32_code  : e = 23 ; break ; // GB_IDIV_SIGNED(y,x,32)"
                case GB_INT64_code  : e = 24 ; break ; // GB_IDIV_SIGNED(y,x,64)"
                case GB_UINT8_code  : e = 25 ; break ; // GB_IDIV_UNSIGNED(y,x,8)"
                case GB_UINT16_code : e = 26 ; break ; // GB_IDIV_UNSIGNED(y,x,16)"
                case GB_UINT32_code : e = 27 ; break ; // GB_IDIV_UNSIGNED(y,x,32)"
                case GB_UINT64_code : e = 28 ; break ; // GB_IDIV_UNSIGNED(y,x,64)"
                default             : e = 29 ; break ; // (y) / (x)
            }
            break ;

        case GB_EQ_opcode :
        case GB_ISEQ_opcode :     // z = (x == y)

            e = 30 ;    // "(x) == (y)" ;
            break ;

        case GB_NE_opcode :
        case GB_ISNE_opcode :     // z = (x != y)

            e = 31 ;    // "(x) != (y)" ;
            break ;

        case GB_GT_opcode :
        case GB_ISGT_opcode :     // z = (x >  y)

            e = 32 ;    // "(x) > (y)" ;
            break ;

        case GB_LT_opcode :
        case GB_ISLT_opcode :     // z = (x <  y)

            e = 33 ;    // "(x) < (y)" ;
            break ;

        case GB_GE_opcode :
        case GB_ISGE_opcode :     // z = (x >= y)

            e = 34 ;    // "(x) >= (y)" ;
            break ;

        case GB_LE_opcode :
        case GB_ISLE_opcode :     // z = (x <= y)

            e = 35 ;    // "(x) <= (y)" ;
            break ;

        case GB_LOR_opcode :      // z = (x != 0) || (y != 0)

            switch (zcode)
            {
                case GB_BOOL_code   : e = 36 ; break ; // "(x) || (y)"
                default             : e = 37 ; break ; // "((x)!=0) || ((y)!=0)"
            }
            break ;

        case GB_LAND_opcode :     // z = (x != 0) && (y != 0)

            switch (zcode)
            {
                case GB_BOOL_code   : e = 38 ; break ; // "(x) && (y)"
                default             : e = 39 ; break ; // "((x)!=0) && ((y)!=0)"
            }
            break ;

        case GB_LXOR_opcode :     // z = (x != 0) != (y != 0)

            switch (zcode)
            {
                case GB_BOOL_code   : e = 40 ; break ; // "(x) != (y)"
                default             : e = 41 ; break ; // "((x)!=0) != ((y)!=0)"
            }
            break ;

        case GB_BOR_opcode       :   // z = (x | y), bitwise or

            if (zcode >= GB_INT8_code && zcode <= GB_UINT64_code)
            {
                e = 42 ;    // "(x) | (y)"
            }
            break ;

        case GB_BAND_opcode      :   // z = (x & y), bitwise and

            if (zcode >= GB_INT8_code && zcode <= GB_UINT64_code)
            {
                e = 43 ;    // "(x) & (y)"
            }
            break ;

        case GB_BXOR_opcode      :   // z = (x ^ y), bitwise xor

            if (zcode >= GB_INT8_code && zcode <= GB_UINT64_code)
            {
                e = 44 ;    // "(x) ^ (y)"
            }
            break ;

        case GB_BXNOR_opcode     :   // z = ~(x ^ y), bitwise xnor

            if (zcode >= GB_INT8_code && zcode <= GB_UINT64_code)
            {
                e = 45 ;    // "~((x) ^ (y))"
            }
            break ;

        case GB_BGET_opcode      :   // z = bitget (x,y)

            switch (zcode)
            {
                case GB_INT8_code   : e = 46 ; break ; // GB_BITGET(x,y,int8_t, 8)
                case GB_INT16_code  : e = 47 ; break ; // GB_BITGET(x,y,int16_t,16)
                case GB_INT32_code  : e = 48 ; break ; // GB_BITGET(x,y,int32_t,32)
                case GB_INT64_code  : e = 49 ; break ; // GB_BITGET(x,y,int64_t,64)
                case GB_UINT8_code  : e = 50 ; break ; // GB_BITGET(x,y,uint8_t,8)
                case GB_UINT16_code : e = 51 ; break ; // GB_BITGET(x,y,uint16_t,16)
                case GB_UINT32_code : e = 52 ; break ; // GB_BITGET(x,y,uint32_t,32)
                case GB_UINT64_code : e = 53 ; break ; // GB_BITGET(x,y,uint64_t,64)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_BSET_opcode      :   // z = bitset (x,y)

            switch (zcode)
            {
                case GB_INT8_code   : e = 54 ; break ; // GB_BITSET(x,y,int8_t, 8)
                case GB_INT16_code  : e = 55 ; break ; // GB_BITSET(x,y,int16_t,16)
                case GB_INT32_code  : e = 56 ; break ; // GB_BITSET(x,y,int32_t,32)
                case GB_INT64_code  : e = 57 ; break ; // GB_BITSET(x,y,int64_t,64)
                case GB_UINT8_code  : e = 58 ; break ; // GB_BITSET(x,y,uint8_t,8)
                case GB_UINT16_code : e = 59 ; break ; // GB_BITSET(x,y,uint16_t,16)
                case GB_UINT32_code : e = 60 ; break ; // GB_BITSET(x,y,uint32_t,32)
                case GB_UINT64_code : e = 61 ; break ; // GB_BITSET(x,y,uint64_t,64)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_BCLR_opcode      :   // z = bitclr (x,y)

            switch (zcode)
            {
                case GB_INT8_code   : e = 62 ; break ; // GB_BITCLR(x,y,int8_t, 8)
                case GB_INT16_code  : e = 63 ; break ; // GB_BITCLR(x,y,int16_t,16)
                case GB_INT32_code  : e = 64 ; break ; // GB_BITCLR(x,y,int32_t,32)
                case GB_INT64_code  : e = 65 ; break ; // GB_BITCLR(x,y,int64_t,64)
                case GB_UINT8_code  : e = 66 ; break ; // GB_BITCLR(x,y,uint8_t,8)
                case GB_UINT16_code : e = 67 ; break ; // GB_BITCLR(x,y,uint16_t,16)
                case GB_UINT32_code : e = 68 ; break ; // GB_BITCLR(x,y,uint32_t,32)
                case GB_UINT64_code : e = 69 ; break ; // GB_BITCLR(x,y,uint64_t,64)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_BSHIFT_opcode    :   // z = bitshift (x,y)

            switch (zcode)
            {
                case GB_INT8_code   : e = 70 ; break ; // GB_bitshift_int8(x,y)
                case GB_INT16_code  : e = 71 ; break ; // GB_bitshift_int16(x,y)
                case GB_INT32_code  : e = 72 ; break ; // GB_bitshift_int32(x,y)
                case GB_INT64_code  : e = 73 ; break ; // GB_bitshift_int64(x,y)
                case GB_UINT8_code  : e = 74 ; break ; // GB_bitshift_uint8(x,y)
                case GB_UINT16_code : e = 75 ; break ; // GB_bitshift_uint16(x,y)
                case GB_UINT32_code : e = 76 ; break ; // GB_bitshift_uint32(x,y)
                case GB_UINT64_code : e = 77 ; break ; // GB_bitshift_uint64(x,y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_POW_opcode    :   // z = pow (x,y)

            switch (zcode)
            {
                case GB_INT8_code   : e = 78 ; break ; // GB_pow_int8 (x, y)
                case GB_INT16_code  : e = 79 ; break ; // GB_pow_int16 (x, y)
                case GB_INT32_code  : e = 80 ; break ; // GB_pow_int32 (x, y)
                case GB_INT64_code  : e = 81 ; break ; // GB_pow_int64 (x, y)
                case GB_UINT8_code  : e = 82 ; break ; // GB_pow_uint8 (x, y)
                case GB_UINT16_code : e = 83 ; break ; // GB_pow_uint16 (x, y)
                case GB_UINT32_code : e = 84 ; break ; // GB_pow_uint32 (x, y)
                case GB_UINT64_code : e = 85 ; break ; // GB_pow_uint64 (x, y)
                case GB_FP32_code   : e = 86 ; break ; // GB_powf (x, y)
                case GB_FP64_code   : e = 87 ; break ; // GB_pow (x, y)
                case GB_FC32_code   : e = 88 ; break ; // GB_cpowf (x, y)
                case GB_FC64_code   : e = 89 ; break ; // GB_cpow (x, y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_ATAN2_opcode     :   // z = atan2 (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 90 ; break ; // atan2f (x, y)
                case GB_FP64_code   : e = 91 ; break ; // atan2 (x, y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_HYPOT_opcode     :   // z = hypot (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 92 ; break ; // hypotf (x, y)
                case GB_FP64_code   : e = 93 ; break ; // hypot (x, y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_FMOD_opcode      :   // z = fmod (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 94 ; break ; // fmodf (x, y)
                case GB_FP64_code   : e = 95 ; break ; // fmod (x, y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_REMAINDER_opcode :   // z = remainder (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 96 ; break ; // remainderf (x, y)
                case GB_FP64_code   : e = 97 ; break ; // remainder (x, y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_COPYSIGN_opcode  :   // z = copysign (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 98 ; break ; // copysignf (x, y)
                case GB_FP64_code   : e = 99 ; break ; // copysign (x, y)
                default             : e = -1 ; break ;
            }
            break ;

        case GB_LDEXP_opcode     :   // z = ldexp (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 100 ; break ; // ldexpf (x, y)
                case GB_FP64_code   : e = 101 ; break ; // ldexp (x, y)
                default             : e = -1  ; break ;
            }
            break ;

        case GB_CMPLX_opcode     :   // z = cmplx (x,y)

            switch (zcode)
            {
                case GB_FP32_code   : e = 102 ; break ; // GxB_CMPLXF (x, y)
                case GB_FP64_code   : e = 103 ; break ; // GxB_CMPLX (x, y)
                default             : e = -1  ; break ;
            }
            break ;

        case GB_PAIR_opcode :       // z = 1

            e = 104 ; break ;       // 1

        case GB_FIRSTI_opcode    :  // z = first_i(A(i,j),y) == i

            e = 105 ; break ;       // z = i

        case GB_FIRSTI1_opcode   :  // z = first_i1(A(i,j),y) == i+1

            e = 106 ; break ;       // z = i+1

        case GB_FIRSTJ_opcode    :  // z = first_j(A(i,j),y) == j

            e = 107 ; break ;       // z = for_semiring ? (k) : (j)

        case GB_FIRSTJ1_opcode   :  // z = first_j1(A(i,j),y) == j+1

            e = 108 ; break ;       // z = for_semiring ? (k+1) : (j+1)

        case GB_SECONDI_opcode   :  // z = second_i(x,B(i,j)) == i

            e = 109 ; break ;       // z = for_semiring ? (k) : (i)

        case GB_SECONDI1_opcode  :  // z = second_i1(x,B(i,j)) == i+1

            e = 110 ; break ;       // z = for_semiring ? (k) : ()

        case GB_SECONDJ_opcode   :  // z = second_j(x,B(i,j)) == j

            e = 111 ; break ;       // z = j

        case GB_SECONDJ1_opcode  :  // z = second_j1(x,B(i,j)) == j+1

            e = 112 ; break ;       // z = j+1

        default : break ;
    }

    (*ecode) = e ;
}

void GB_cuda_charify_binop
(
    // output:
    char **op_string,   // string defining the operator
    // input:
    int ecode,          // from GB_cuda_enumify_binop
    bool for_semiring   // true for A*B, false for A+B or A.*B (not needed)
)
{
    const char *f ;

    switch (ecode)
    {

        // first
        case 0 : f = "(x)"              ; break ;

        // any, second
        case 1 : f = "(y)"              ; break ;

        // min
        case 2 : f = "fminf (x,y)"      ; break ;
        case 3 : f = "fmin (x,y)"       ; break ;
        case 4 : f = "GB_IMIN (x,y)"    ; break ;

        // max
        case 5 : f = "fmaxf (x,y)"      ; break ;
        case 6 : f = "fmax (x,y)"       ; break ;
        case 7 : f = "GB_IMAX (x,y)"    ; break ;

        // plus
        case 8 : f = "(x) + (y)"        ; break ;

        // minus
        case 9 : f = "(x) - (y)"        ; break ;

        // rminus
        case 10 : f = "(y) - (x)"       ; break ;

        // times
        case 11 : f = "(x) * (y)"       ; break ;

        // div
        case 12 : f = "GB_IDIV_SIGNED(x,y,8)"           ; break ;
        case 13 : f = "GB_IDIV_SIGNED(x,y,16)"          ; break ;
        case 14 : f = "GB_IDIV_SIGNED(x,y,32)"          ; break ;
        case 15 : f = "GB_IDIV_SIGNED(x,y,64)"          ; break ;
        case 16 : f = "GB_IDIV_UNSIGNED(x,y,8)"         ; break ;
        case 17 : f = "GB_IDIV_UNSIGNED(x,y,16)"        ; break ;
        case 18 : f = "GB_IDIV_UNSIGNED(x,y,32)"        ; break ;
        case 19 : f = "GB_IDIV_UNSIGNED(x,y,64)"        ; break ;
        case 20 : f = "(x) / (y)"                       ; break ;

        // rdiv
        case 21 : f = "GB_IDIV_SIGNED(y,x,8)"           ; break ;
        case 22 : f = "GB_IDIV_SIGNED(y,x,16)"          ; break ;
        case 23 : f = "GB_IDIV_SIGNED(y,x,32)"          ; break ;
        case 24 : f = "GB_IDIV_SIGNED(y,x,64)"          ; break ;
        case 25 : f = "GB_IDIV_UNSIGNED(y,x,8)"         ; break ;
        case 26 : f = "GB_IDIV_UNSIGNED(y,x,16)"        ; break ;
        case 27 : f = "GB_IDIV_UNSIGNED(y,x,32)"        ; break ;
        case 28 : f = "GB_IDIV_UNSIGNED(y,x,64)"        ; break ;
        case 29 : f = "(y) / (x)"                       ; break ;

        // eq, iseq
        case 30 : f = "(x) == (y)"                      ; break ;

        // ne, isne
        case 31 : f = "(x) != (y)"                      ; break ;

        // gt, isgt
        case 32 : f = "(x) > (y)"                       ; break ;

        // lt, islt
        case 33 : f = "(x) < (y)"                       ; break ;

        // ge, isget
        case 34 : f = "(x) >= (y)"                      ; break ;

        // le, isle
        case 35 : f = "(x) <= (y)"                      ; break ;

        // lor
        case 36 : f = "(x) || (y)"                      ; break ;
        case 37 : f = "((x)!=0) || ((y)!=0)"            ; break ;

        // land
        case 38 : f = "(x) && (y)"                      ; break ;
        case 39 : f = "((x)!=0) && ((y)!=0)"            ; break ;

        // lxor
        case 40 : f = "(x) != (y)"                      ; break ;
        case 41 : f = "((x)!=0) != ((y)!=0)"            ; break ;

        // bor
        case 42 : f = "(x) | (y)"                       ; break ;

        // band
        case 43 : f = "(x) & (y)"                       ; break ;

        // bxor
        case 44 : f = "(x) ^ (y)"                       ; break ;

        // bxnor
        case 45 : f = "~((x) ^ (y))"                    ; break ;

        // bget
        case 46 : f = "GB_BITGET(x,y,int8_t, 8)"        ; break ;
        case 47 : f = "GB_BITGET(x,y,int16_t,16)"       ; break ;
        case 48 : f = "GB_BITGET(x,y,int32_t,32)"       ; break ;
        case 49 : f = "GB_BITGET(x,y,int64_t,64)"       ; break ;
        case 50 : f = "GB_BITGET(x,y,uint8_t,8)"        ; break ;
        case 51 : f = "GB_BITGET(x,y,uint16_t,16)"      ; break ;
        case 52 : f = "GB_BITGET(x,y,uint32_t,32)"      ; break ;
        case 53 : f = "GB_BITGET(x,y,uint64_t,64)"      ; break ;

        // bset
        case 54 : f = "GB_BITSET(x,y,int8_t, 8)"        ; break ;
        case 55 : f = "GB_BITSET(x,y,int16_t,16)"       ; break ;
        case 56 : f = "GB_BITSET(x,y,int32_t,32)"       ; break ;
        case 57 : f = "GB_BITSET(x,y,int64_t,64)"       ; break ;
        case 58 : f = "GB_BITSET(x,y,uint8_t,8)"        ; break ;
        case 59 : f = "GB_BITSET(x,y,uint16_t,16)"      ; break ;
        case 60 : f = "GB_BITSET(x,y,uint32_t,32)"      ; break ;
        case 61 : f = "GB_BITSET(x,y,uint64_t,64)"      ; break ;

        // bclr
        case 62 : f = "GB_BITCLR(x,y,int8_t, 8)"        ; break ;
        case 63 : f = "GB_BITCLR(x,y,int16_t,16)"       ; break ;
        case 64 : f = "GB_BITCLR(x,y,int32_t,32)"       ; break ;
        case 65 : f = "GB_BITCLR(x,y,int64_t,64)"       ; break ;
        case 66 : f = "GB_BITCLR(x,y,uint8_t,8)"        ; break ;
        case 67 : f = "GB_BITCLR(x,y,uint16_t,16)"      ; break ;
        case 68 : f = "GB_BITCLR(x,y,uint32_t,32)"      ; break ;
        case 69 : f = "GB_BITCLR(x,y,uint64_t,64)"      ; break ;

        // bshift
        case 70 : f = "GB_bitshift_int8(x,y)"           ; break ;
        case 71 : f = "GB_bitshift_int16(x,y)"          ; break ;
        case 72 : f = "GB_bitshift_int32(x,y)"          ; break ;
        case 73 : f = "GB_bitshift_int64(x,y)"          ; break ;
        case 74 : f = "GB_bitshift_uint8(x,y)"          ; break ;
        case 75 : f = "GB_bitshift_uint16(x,y)"         ; break ;
        case 76 : f = "GB_bitshift_uint32(x,y)"         ; break ;
        case 77 : f = "GB_bitshift_uint64(x,y)"         ; break ;

        // pow
        case 78 : f = "GB_pow_int8 (x, y)"              ; break ;
        case 79 : f = "GB_pow_int16 (x, y)"             ; break ;
        case 80 : f = "GB_pow_int32 (x, y)"             ; break ;
        case 81 : f = "GB_pow_int64 (x, y)"             ; break ;
        case 82 : f = "GB_pow_uint8 (x, y)"             ; break ;
        case 83 : f = "GB_pow_uint16 (x, y)"            ; break ;
        case 84 : f = "GB_pow_uint32 (x, y)"            ; break ;
        case 85 : f = "GB_pow_uint64 (x, y)"            ; break ;
        case 86 : f = "GB_powf (x, y)"                  ; break ;
        case 87 : f = "GB_pow (x, y)"                   ; break ;
        case 88 : f = "GB_cpowf (x, y)"                 ; break ;
        case 89 : f = "GB_cpow (x, y)"                  ; break ;

        // atan2
        case 90 : f = "atan2f (x, y)"                   ; break ;
        case 91 : f = "atan2 (x, y)"                    ; break ;

        // hypot
        case 92 : f = "hypotf (x, y)"                   ; break ;
        case 93 : f = "hypot (x, y)"                    ; break ;

        // fmod
        case 94 : f = "fmodf (x, y)"                    ; break ;
        case 95 : f = "fmod (x, y)"                     ; break ;

        // remainder
        case 96 : f = "remainderf (x, y)"               ; break ;
        case 97 : f = "remainder (x, y)"                ; break ;

        // copysign
        case 98 : f = "copysignf (x, y)"                ; break ;
        case 99 : f = "copysign (x, y)"                 ; break ;

        // ldexp
        case 100 : f = "ldexpf (x, y)"                  ; break ;
        case 101 ; f = "ldexp (x, y)"                   ; break ;

        // cmplex
        case 102 : f = "GxB_CMPLXF (x, y)"              ; break ;
        case 103 : f = "GxB_CMPLX (x, y)"               ; break ;

        // pair
        case 104 : f = "(1)"                            ; break ;

        // firsti
        case 105 : f = "(i)"                            ; break ;

        // firsti1
        case 106 : f = "(i+1)"                          ; break ;

        // firstj
        case 107 : f = for_semiring ? "(k)" : "(j)"     ; break ;

        // firstj1
        case 108 : f = for_semiring ? "(k+1)" : "(j+1)" ; break ;

        // secondi
        case 109 : f = for_semiring ? "(k)" : "(i)"     ; break ;

        // secondi1
        case 110 : f = for_semiring ? "(k+1)" : "(i+1)" ; break ;

        // secondj
        case 111 : f = "(j)"                            ; break ;

        // secondj1
        case 112 : f = "(j+1)"                          ; break ;

        default : f = NULL ;                            ; break ;
    }

    (*op_string) = f ;
}

void GB_cuda_macrofy_binop
(
    // output:
    char *code_string,  // string with the #define macro
    // input:
    const char *macro_name,   // name of macro to construct
    char *op_string     // string defining the operator
)
{
    snprintf (code_string, GB_CUDA_STRLEN,
        "#define %s(x,y) (%s)", macro_name, op_string) ;
}

