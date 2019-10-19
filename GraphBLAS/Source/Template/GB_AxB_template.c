//------------------------------------------------------------------------------
// GB_AxB_template.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_builtin.c

// This file is used for 17 operators, which is #defined as IMULT(x,y) and
// FMULT(x,y) by the including file.  IMULT(x,y) is used for integers and
// FMULT(x,y) for floating-point.  The multiply operator is combined here
// with 40 or 44 monoids to create 40 or 44 unique semiring workers.

//      FIRST, SECOND, MIN, MAX, PLUS, MINUS, TIMES, DIV,
//      ISEQ, ISNE, ISGT, ISLT, ISGE, ISLE,
//      LAND, LOR, LXOR.

// For all of them, the types of x, y, and z are the same.
// There are 40 non-boolean monoids and 4 boolean monoids defined here.

// NO_BOOLEAN is #defined for 12 of these multiply operators in the
// #include'ing file, GB_AxB_builtin.c (min, max, plus, minus, times, div, is*)
// since those 12 multiply operators are redundant and have been renamed.  For
// these 12, the boolean monoids are not needed.

ASSERT (zcode == xycode) ;

if (zcode != GB_BOOL_code)
{
    switch (add_opcode)
    {

        case GB_MIN_opcode     :   // w = min (w,t), identity is +inf

            switch (zcode)
            {
                //   zcode                 ztype     xytype    identity
                #define ADD(w,t)  w = IMIN (w,t)
                #define MULT(x,y) IMULT(x,y)
                case GB_INT8_code   : AxB (int8_t  , int8_t  , INT8_MAX)   ;
                case GB_UINT8_code  : AxB (uint8_t , uint8_t , UINT8_MAX)  ;
                case GB_INT16_code  : AxB (int16_t , int16_t , INT16_MAX)  ;
                case GB_UINT16_code : AxB (uint16_t, uint16_t, UINT16_MAX) ;
                case GB_INT32_code  : AxB (int32_t , int32_t , INT32_MAX)  ;
                case GB_UINT32_code : AxB (uint32_t, uint32_t, UINT32_MAX) ;
                case GB_INT64_code  : AxB (int64_t , int64_t , INT64_MAX)  ;
                case GB_UINT64_code : AxB (uint64_t, uint64_t, UINT64_MAX) ;
                #undef  ADD
                #undef  MULT
                #define ADD(w,t) w = FMIN (w,t)
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (float   , float   , INFINITY)   ;
                case GB_FP64_code   : AxB (double  , double  , INFINITY)   ;
                #undef  ADD
                #undef  MULT
                default: ;
            }
            break ;

        case GB_MAX_opcode     :   // w = max (w,t), identity is -inf

            switch (zcode)
            {
                //   zcode                 ztype     xytype    identity
                #define ADD(w,t) w = IMAX (w,t)
                #define MULT(x,y) IMULT(x,y)
                case GB_INT8_code   : AxB (int8_t  , int8_t  , INT8_MIN)   ;
                case GB_UINT8_code  : AxB (uint8_t , uint8_t , 0)          ;
                case GB_INT16_code  : AxB (int16_t , int16_t , INT16_MIN)  ;
                case GB_UINT16_code : AxB (uint16_t, uint16_t, 0)          ;
                case GB_INT32_code  : AxB (int32_t , int32_t , INT32_MIN)  ;
                case GB_UINT32_code : AxB (uint32_t, uint32_t, 0)          ;
                case GB_INT64_code  : AxB (int64_t , int64_t , INT64_MIN)  ;
                case GB_UINT64_code : AxB (uint64_t, uint64_t, 0)          ;
                #undef  ADD
                #undef  MULT
                #define ADD(w,t) w = FMAX (w,t)
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (float   , float   , -INFINITY)  ;
                case GB_FP64_code   : AxB (double  , double  , -INFINITY)  ;
                #undef  ADD
                #undef  MULT
                default: ;
            }
            break ;

        case GB_PLUS_opcode    :   // w += t, identity is 0

            #define ADD(w,t) w += t
            switch (zcode)
            {
                //   zcode                 ztype     xytype    identity
                #define MULT(x,y) IMULT(x,y)
                case GB_INT8_code   : AxB (int8_t  , int8_t  , 0)          ;
                case GB_UINT8_code  : AxB (uint8_t , uint8_t , 0)          ;
                case GB_INT16_code  : AxB (int16_t , int16_t , 0)          ;
                case GB_UINT16_code : AxB (uint16_t, uint16_t, 0)          ;
                case GB_INT32_code  : AxB (int32_t , int32_t , 0)          ;
                case GB_UINT32_code : AxB (uint32_t, uint32_t, 0)          ;
                case GB_INT64_code  : AxB (int64_t , int64_t , 0)          ;
                case GB_UINT64_code : AxB (uint64_t, uint64_t, 0)          ;
                #undef  MULT
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (float   , float   , 0)          ;
                case GB_FP64_code   : AxB (double  , double  , 0)          ;
                #undef  MULT
                default: ;
            }
            break ;
            #undef  ADD

        case GB_TIMES_opcode   :   // w *= t, identity is 1

            #define ADD(w,t) w *= t
            switch (zcode)
            {
                //   zcode                 ztype     xytype    identity
                #define MULT(x,y) IMULT(x,y)
                case GB_INT8_code   : AxB (int8_t  , int8_t  , 1)          ;
                case GB_UINT8_code  : AxB (uint8_t , uint8_t , 1)          ;
                case GB_INT16_code  : AxB (int16_t , int16_t , 1)          ;
                case GB_UINT16_code : AxB (uint16_t, uint16_t, 1)          ;
                case GB_INT32_code  : AxB (int32_t , int32_t , 1)          ;
                case GB_UINT32_code : AxB (uint32_t, uint32_t, 1)          ;
                case GB_INT64_code  : AxB (int64_t , int64_t , 1)          ;
                case GB_UINT64_code : AxB (uint64_t, uint64_t, 1)          ;
                #undef  MULT
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (float   , float   , 1)          ;
                case GB_FP64_code   : AxB (double  , double  , 1)          ;
                #undef  MULT
                default: ;
            }
            break ;
            #undef  ADD

        default: ;
    }

}

#ifndef NO_BOOLEAN
else
{

        #define MULT(x,y) IMULT(x,y)
        switch (add_opcode)
        {

            case GB_LOR_opcode     :

                #define ADD(w,t) w = (w || t)
                //   ztype xytype identity
                AxB (bool, bool,  false) ;
                #undef  ADD

            case GB_LAND_opcode    :

                #define ADD(w,t) w = (w && t)
                //   ztype xytype identity
                AxB (bool, bool,  true) ;
                #undef  ADD

            case GB_LXOR_opcode    :

                #define ADD(w,t) w = (w != t)
                //   ztype xytype identity
                AxB (bool, bool,  false) ;
                #undef  ADD

            case GB_EQ_opcode    :

                #define ADD(w,t) w = (w == t)
                //   ztype xytype identity
                AxB (bool, bool,  true) ;
                #undef  ADD

            default: ;
        }
        #undef  MULT

}
#endif

#undef NO_BOOLEAN
#undef MULT
#undef IMULT
#undef FMULT

