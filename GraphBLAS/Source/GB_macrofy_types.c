//------------------------------------------------------------------------------
// GB_macrofy_types: construct typedefs for up to 6 types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_types
(
    FILE *fp,
    // input:
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype,
    GrB_Type xtype,
    GrB_Type ytype,
    GrB_Type ztype
)
{

    //--------------------------------------------------------------------------
    // define complex types, if any types are GxB_FC32 or GxB_FC64
    //--------------------------------------------------------------------------

    // FIXME: this is only a partial solution.  The user-defined types and
    // operators may wish to use the C/C++ complex types, or the GraphBLAS
    // typedefs GxB_FC32_t and GxB_FC64_t.  See Demo/Source/usercomplex.c.

    // Solution: each CUDA and CPU JIT kernel must always define GxB_FC32_t
    // and GxB_FC64_t.  Then this block of code can be deleted.

    if (ctype == GxB_FC32 || ctype == GxB_FC64 ||
        atype == GxB_FC32 || atype == GxB_FC64 ||
        btype == GxB_FC32 || btype == GxB_FC64 ||
        xtype == GxB_FC32 || xtype == GxB_FC64 ||
        ytype == GxB_FC32 || ytype == GxB_FC64 ||
        ztype == GxB_FC32 || ztype == GxB_FC64)
    {

        fprintf (fp, "#ifndef GB_COMPLEX_H\n"
                     "#define GB_COMPLEX_H\n") ;

        #if GB_COMPILER_MSC

            // CPU only, since Windows doesn't support the
            // unified shared memory model of CUDA.
            fprintf (fp,
               "    // Microsoft Windows complex types              \n"
               "    #include <complex.h>                            \n"
               "    #undef I                                        \n"
               "    typedef _Fcomplex GxB_FC32_t ;                  \n"
               "    typedef _Dcomplex GxB_FC64_t ;                  \n"
               "    #define GxB_CMPLXF(r,i) (_FCbuild (r,i))        \n"
               "    #define GxB_CMPLX(r,i)  ( _Cbuild (r,i))        \n") ;

        #else

            // for the CPU (in C) and GPU (CUDA, using C++)
            fprintf (fp,
               "    // complex types                                    \n"
               "    #if defined ( __NVCC__ )                            \n"
               "        // CUDA complex types (using the C++ variant)   \n"
               "        #include <cuda/std/complex>                     \n"
               "        typedef std::complex<float>  GxB_FC32_t ;       \n"
               "        typedef std::complex<double> GxB_FC64_t ;       \n"
               "        #define GxB_CMPLXF(r,i) GxB_FC32_t(r,i)         \n"
               "        #define GxB_CMPLX(r,i)  GxB_FC64_t(r,i)         \n"
               "    #elif defined ( __cplusplus )                       \n"
               "        // C++ complex types                            \n"
               "        extern \"C++\"                                  \n"
               "        {                                               \n"
               "            #include <cmath>                            \n"
               "            #include <complex>                          \n"
               "            typedef std::complex<float>  GxB_FC32_t ;   \n"
               "            typedef std::complex<double> GxB_FC64_t ;   \n"
               "        }                                               \n"
               "        #define GxB_CMPLXF(r,i) GxB_FC32_t(r,i)         \n"
               "        #define GxB_CMPLX(r,i)  GxB_FC64_t(r,i)         \n"
               "    #else                                               \n"
               "        // ANSI C11 complex types                       \n"
               "        #include <complex.h>                            \n"
               "        typedef float  complex GxB_FC32_t ;             \n"
               "        typedef double complex GxB_FC64_t ;             \n"
               "        #define GxB_CMPLX(r,i)  CMPLX (r,i)             \n"
               "        #define GxB_CMPLXF(r,i) CMPLXF (r,i)            \n"
               "    #endif                                              \n"
               "    #undef I                                            \n") ;

        #endif

        fprintf (fp, "#endif\n\n") ;
    }

    //--------------------------------------------------------------------------
    // create typedefs, checking for duplicates
    //--------------------------------------------------------------------------

    const char *defn [6] ;
    defn [0] = (ctype == NULL) ? NULL : ctype->defn ;
    defn [1] = (atype == NULL) ? NULL : atype->defn ;
    defn [2] = (btype == NULL) ? NULL : btype->defn ;
    defn [3] = (xtype == NULL) ? NULL : xtype->defn ;
    defn [4] = (ytype == NULL) ? NULL : ytype->defn ;
    defn [5] = (ztype == NULL) ? NULL : ztype->defn ;

    for (int k = 0 ; k <= 5 ; k++)
    {
        if (defn [k] != NULL)
        {
            // only print this typedef it is unique
            bool is_unique = true ;
            for (int j = 0 ; j < k && is_unique ; j++)
            {
                if (defn [j] != NULL && strcmp (defn [j], defn [k]) == 0)
                {
                    is_unique = false ;
                }
            }
            if (is_unique)
            {
                // the typedef is unique: include it in the .h file
                fprintf (fp, "%s\n\n", defn [k]) ;
            }
        }
    }
}

