//------------------------------------------------------------------------------
// GB_macrofy_cast: construct a macro and defn for typecasting
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

/*
    // the 14 scalar types: 13 built-in types, and one user-defined type code
    GB_ignore_code  = 0,
    GB_BOOL_code    = 1,        // 'logical' in @GrB interface
    GB_INT8_code    = 2,
    GB_UINT8_code   = 3,
    GB_INT16_code   = 4,
    GB_UINT16_code  = 5,
    GB_INT32_code   = 6,
    GB_UINT32_code  = 7,
    GB_INT64_code   = 8,
    GB_UINT64_code  = 9,
    GB_FP32_code    = 10,       // float ('single' in @GrB interface)
    GB_FP64_code    = 11,       // double
    GB_FC32_code    = 12,       // float complex ('single complex' in @GrB)
    GB_FC64_code    = 13,       // double complex
    GB_UDT_code     = 14        // void *, user-defined type
*/

// constructs a macro of the form:

//      #define macro(z,x...) z = ...

// The name of the macro is given by macro_name.
// The name of the variable z is given by zarg.
// The inputs x,... are given by xargs (which may containt a comma).
// xexpr is an expression using the xargs that produce a value of type xtype.
// z has type ztype.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_cast
(
    FILE *fp,
    // input:
    const char *macro_name,     // name of the macro: #define macro(z,x...)
    const char *zarg,           // name of the z argument of the macro
    const char *xargs,          // one or more x arguments
    const char *xexpr,          // an expression based on xargs
    const GrB_Type ztype,       // the type of the z output
    const GrB_Type xtype        // the type of the x input
)
{

    const GB_Type_code zcode = ztype->code ;
    const GB_Type_code xcode = xtype->code ;
    const char *f ;
    int args = 2 ;

    if (zcode == xcode)
    {

        //----------------------------------------------------------------------
        // no typecasting
        //----------------------------------------------------------------------

        // user-defined types come here, and require type->defn and type->name
        // to both be defined.  If the user-defined type has no name or defn,
        // then no JIT kernel can be created for it.

        ASSERT (GB_IMPLIES (zcode == GB_UDT_code, (ztype == xtype) &&
            ztype->name != NULL && ztype->defn != NULL)) ;

        f = "%s = (%s)" ;

    }
    else if (zcode == GB_BOOL_code)
    {

        //----------------------------------------------------------------------
        // typecast to boolean
        //----------------------------------------------------------------------

        if (zcode == GB_FC32_code)
        {
            f = "%s = (crealf (%s) != 0 || cimagf (%s) != 0)" ;
            args = 3 ;
        }
        else if (zcode == GB_FC64_code)
        {
            f = "%s = (creal (%s) != 0 || cimag (%s) != 0)" ;
            args = 3 ;
        }
        else
        {
            f = "%s = ((%s) != 0)" ;
        }

    }
    else if ((zcode >= GB_INT8_code && zcode <= GB_UINT64_code)
          && (xcode >= GB_FP32_code && zcode <= GB_FC64_code))
    {

        //----------------------------------------------------------------------
        // typecast to integer from floating-point
        //----------------------------------------------------------------------

        switch (zcode)
        {
            case GB_INT8_code   : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_int8_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_int8_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_int8_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_int8_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_INT8",
                    GB_CAST_TO_INT8_DEFN) ;
                break ;

            case GB_INT16_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_int16_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_int16_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_int16_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_int16_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_INT16",
                    GB_CAST_TO_INT16_DEFN) ;
                break ;

            case GB_INT32_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_int32_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_int32_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_int32_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_int32_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_INT32",
                    GB_CAST_TO_INT32_DEFN) ;
                break ;

            case GB_INT64_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_int64_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_int64_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_int64_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_int64_t (creal (x))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_INT64",
                    GB_CAST_TO_INT64_DEFN) ;
                break ;

            case GB_UINT8_code  : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_uint8_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_uint8_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_uint8_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_uint8_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_UINT8",
                    GB_CAST_TO_UINT8_DEFN) ;
                break ;

            case GB_UINT16_code : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_uint16_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_uint16_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_uint16_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_uint16_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_UINT16",
                    GB_CAST_TO_UINT16_DEFN) ;
                break ;

            case GB_UINT32_code : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_uint32_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_uint32_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_uint32_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_uint32_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_UINT32",
                    GB_CAST_TO_UINT32_DEFN) ;
                break ;

            case GB_UINT64_code : 

                switch (xcode)
                {
                    case GB_FP32_code : 
                        f = "%s = GB_cast_to_uint64_t ((double) (%s))" ;
                        break ;
                    case GB_FP64_code : 
                        f = "%s = GB_cast_to_uint64_t (%s)" ;
                        break ;
                    case GB_FC32_code : 
                        f = "%s = GB_cast_to_uint64_t ((double) crealf (%s))" ;
                        break ;
                    case GB_FC64_code : 
                        f = "%s = GB_cast_to_uint64_t (creal (%s))" ;
                        break ;
                    default:;
                }
                GB_macrofy_defn (fp, 0, "GB_CAST_TO_UINT64",
                    GB_CAST_TO_UINT64_DEFN) ;
                break ;

            default:;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // all other cases: use ANSI C11 typecasting rules
        //----------------------------------------------------------------------

        f = NULL ;
        fprintf (fp, "#define %s(%s,%s) %s = ((%s) (%s))\n",
            macro_name, zarg, xargs, zarg, ztype->name, xexpr) ;
    }

    if (f != NULL)
    {
        #define SLEN 1024
        char s [SLEN+1] ;
        if (args == 3)
        {
            snprintf (s, SLEN, f, zarg, xexpr, xexpr) ;
        }
        else
        {
            snprintf (s, SLEN, f, zarg, xexpr) ;
        }
        fprintf (fp, "#define %s(%s,%s) %s\n", macro_name, zarg, xargs, s) ;
    }
}

