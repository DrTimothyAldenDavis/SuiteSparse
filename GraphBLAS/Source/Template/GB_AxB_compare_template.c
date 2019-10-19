//------------------------------------------------------------------------------
// GB_AxB_compare_template.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_builtin.c

// The multiply operator is a comparator: EQ, NE, GT, LT, GE, LE.
// z=f(x,y): x and x are either boolean or non-boolean.  z is boolean.

// Since z is boolean, the only monoids available are OR, AND, XOR, and EQ
// All the other four (max==plus==or, min==times==and) are redundant.
// Those opcodes have been renamed, and handled by the OR and AND workers
// defined here.

// There is one special case to consider.  For boolean x, y, and z, the
// function z=NE(x,y) is the same as z=XOR(x,y).  If z is boolean, the multiply
// operator NE has already been renamed XOR by GB_AxB_builtin, and thus NE will
// never use the boolean case, below.  Thus it is removed with the #ifndef
// NO_BOOLEAN.

ASSERT (zcode == GB_BOOL_code) ;
{

    // C = A*B where C is boolean, but A and B are non-boolean.
    // The result of the compare(A,B) operation is boolean.
    // There are 4 monoids available: OR, AND, XOR, EQ

    switch (add_opcode)
    {

        case GB_LOR_opcode     :

            #define ADD(w,t) w = (w || t)
            switch (xycode)
            {
                //   xycode                ztype     xytype    identity
                #define MULT(x,y) IMULT(x,y)
                #ifndef NO_BOOLEAN
                case GB_BOOL_code   : AxB (bool    , bool    , false)      ;
                #endif
                case GB_INT8_code   : AxB (bool    , int8_t  , false)      ;
                case GB_UINT8_code  : AxB (bool    , uint8_t , false)      ;
                case GB_INT16_code  : AxB (bool    , int16_t , false)      ;
                case GB_UINT16_code : AxB (bool    , uint16_t, false)      ;
                case GB_INT32_code  : AxB (bool    , int32_t , false)      ;
                case GB_UINT32_code : AxB (bool    , uint32_t, false)      ;
                case GB_INT64_code  : AxB (bool    , int64_t , false)      ;
                case GB_UINT64_code : AxB (bool    , uint64_t, false)      ;
                #undef  MULT
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (bool    , float   , false)      ;
                case GB_FP64_code   : AxB (bool    , double  , false)      ;
                #undef  MULT
                default: ;
            }
            break ;
            #undef  ADD

        case GB_LAND_opcode    :

            #define ADD(w,t) w = (w && t)
            switch (xycode)
            {
                //   xycode                ztype     xytype    identity
                #define MULT(x,y) IMULT(x,y)
                #ifndef NO_BOOLEAN
                case GB_BOOL_code   : AxB (bool    , bool    , true)       ;
                #endif
                case GB_INT8_code   : AxB (bool    , int8_t  , true)       ;
                case GB_UINT8_code  : AxB (bool    , uint8_t , true)       ;
                case GB_INT16_code  : AxB (bool    , int16_t , true)       ;
                case GB_UINT16_code : AxB (bool    , uint16_t, true)       ;
                case GB_INT32_code  : AxB (bool    , int32_t , true)       ;
                case GB_UINT32_code : AxB (bool    , uint32_t, true)       ;
                case GB_INT64_code  : AxB (bool    , int64_t , true)       ;
                case GB_UINT64_code : AxB (bool    , uint64_t, true)       ;
                #undef  MULT
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (bool    , float   , true)       ;
                case GB_FP64_code   : AxB (bool    , double  , true)       ;
                #undef  MULT
                default: ;
            }
            break ;
            #undef  ADD

        case GB_LXOR_opcode    :

            #define ADD(w,t) w = (w != t)
            switch (xycode)
            {
                //   xycode                ztype     xytype    identity
                #define MULT(x,y) IMULT(x,y)
                #ifndef NO_BOOLEAN
                case GB_BOOL_code   : AxB (bool    , bool    , false)      ;
                #endif
                case GB_INT8_code   : AxB (bool    , int8_t  , false)      ;
                case GB_UINT8_code  : AxB (bool    , uint8_t , false)      ;
                case GB_INT16_code  : AxB (bool    , int16_t , false)      ;
                case GB_UINT16_code : AxB (bool    , uint16_t, false)      ;
                case GB_INT32_code  : AxB (bool    , int32_t , false)      ;
                case GB_UINT32_code : AxB (bool    , uint32_t, false)      ;
                case GB_INT64_code  : AxB (bool    , int64_t , false)      ;
                case GB_UINT64_code : AxB (bool    , uint64_t, false)      ;
                #undef  MULT
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (bool    , float   , false)      ;
                case GB_FP64_code   : AxB (bool    , double  , false)      ;
                #undef  MULT
                default: ;
            }
            break ;
            #undef  ADD

        case GB_EQ_opcode    :

            #define ADD(w,t) w = (w == t)
            switch (xycode)
            {
                //   xycode                ztype     xytype    identity
                #define MULT(x,y) IMULT(x,y)
                #ifndef NO_BOOLEAN
                case GB_BOOL_code   : AxB (bool    , bool    , true)       ;
                #endif
                case GB_INT8_code   : AxB (bool    , int8_t  , true)       ;
                case GB_UINT8_code  : AxB (bool    , uint8_t , true)       ;
                case GB_INT16_code  : AxB (bool    , int16_t , true)       ;
                case GB_UINT16_code : AxB (bool    , uint16_t, true)       ;
                case GB_INT32_code  : AxB (bool    , int32_t , true)       ;
                case GB_UINT32_code : AxB (bool    , uint32_t, true)       ;
                case GB_INT64_code  : AxB (bool    , int64_t , true)       ;
                case GB_UINT64_code : AxB (bool    , uint64_t, true)       ;
                #undef  MULT
                #define MULT(x,y) FMULT(x,y)
                case GB_FP32_code   : AxB (bool    , float   , true)       ;
                case GB_FP64_code   : AxB (bool    , double  , true)       ;
                #undef  MULT
                default: ;
            }
            break ;
            #undef  ADD

        default: ;
    }
}

#undef NO_BOOLEAN
#undef MULT
#undef IMULT
#undef FMULT

