//------------------------------------------------------------------------------
// GB_AxB_bitwise_factory.c: switch factory for C=A*B (bitwise monoids)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_factory.c, which calls up to 16
// bitwise semirings.

{
    switch (add_opcode)
    {

        //----------------------------------------------------------------------
        case GB_BOR_opcode :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_bor, GB_MULT_NAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_bor, GB_MULT_NAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_bor, GB_MULT_NAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_bor, GB_MULT_NAME, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_opcode :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_band, GB_MULT_NAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_band, GB_MULT_NAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_band, GB_MULT_NAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_band, GB_MULT_NAME, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_opcode :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_bxor, GB_MULT_NAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_bxor, GB_MULT_NAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_bxor, GB_MULT_NAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_bxor, GB_MULT_NAME, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_opcode :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_bxnor, GB_MULT_NAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_bxnor, GB_MULT_NAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_bxnor, GB_MULT_NAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_bxnor, GB_MULT_NAME, _uint64)
                default: ;
            }
            break ;

        default: ;
    }
}

#undef GB_MULT_NAME

