//------------------------------------------------------------------------------
// GxB_Global_Option_set: set a global default option for all future matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Global_Option_set      // set a global default option
(
    GxB_Option_Field field,         // option to change
    ...                             // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GxB_Global_Option_set (field, value)") ;

    //--------------------------------------------------------------------------
    // set the global option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        //----------------------------------------------------------------------
        // matrix format
        //----------------------------------------------------------------------

        case GxB_HYPER : 

            { 
                va_start (ap, field) ;
                double hyper_ratio = va_arg (ap, double) ;
                va_end (ap) ;
                GB_Global_hyper_ratio_set (hyper_ratio) ;
            }
            break ;

        case GxB_FORMAT : 

            { 
                va_start (ap, field) ;
                int format = va_arg (ap, int) ;
                va_end (ap) ;
                if (! (format == GxB_BY_ROW || format == GxB_BY_COL))
                { 
                    return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,
                            "unsupported format [%d], must be one of:\n"
                            "GxB_BY_ROW [%d] or GxB_BY_COL [%d]", format,
                            (int) GxB_BY_ROW, (int) GxB_BY_COL))) ;
                }
                GB_Global_is_csc_set (format != (int) GxB_BY_ROW) ; 
            }
            break ;

        //----------------------------------------------------------------------
        // OpenMP control
        //----------------------------------------------------------------------

        case GxB_GLOBAL_NTHREADS :      // same as GxB_NTHREADS

            { 
                va_start (ap, field) ;
                int nthreads_max_new = va_arg (ap, int) ;
                va_end (ap) ;
                // if < 1, then treat it as if nthreads_max = 1
                nthreads_max_new = GB_IMAX (1, nthreads_max_new) ;
                GB_Global_nthreads_max_set (nthreads_max_new) ;
            }
            break ;

        case GxB_GLOBAL_CHUNK :         // same as GxB_CHUNK

            { 
                va_start (ap, field) ;
                double chunk = va_arg (ap, double) ;
                va_end (ap) ;
                GB_Global_chunk_set (chunk) ;
            }
            break ;

        //----------------------------------------------------------------------
        // diagnostics
        //----------------------------------------------------------------------

        case GxB_BURBLE :

            { 
                va_start (ap, field) ;
                int burble = va_arg (ap, int) ;
                va_end (ap) ;
                GB_Global_burble_set ((bool) burble) ;
            }
            break ;

        //----------------------------------------------------------------------
        // CUDA (in progress)
        //----------------------------------------------------------------------

        case GxB_GLOBAL_GPU_CONTROL :       // same as GxB_GPU_CONTROL

            { 
                va_start (ap, field) ;
                GrB_Desc_Value gpu_control = (GrB_Desc_Value) va_arg (ap, int) ;
                va_end (ap) ;
                GB_Global_gpu_control_set (gpu_control) ;
            }
            break ;

        case GxB_GLOBAL_GPU_CHUNK :         // same as GxB_GPU_CHUNK

            { 
                va_start (ap, field) ;
                double gpu_chunk = va_arg (ap, double) ;
                va_end (ap) ;
                GB_Global_gpu_chunk_set (gpu_chunk) ;
            }
            break ;

        //----------------------------------------------------------------------
        // Intel MKL (in progress)
        //----------------------------------------------------------------------

        case GxB_GLOBAL_MKL :          // same as GxB_MKL

            { 
                va_start (ap, field) ;
                int use_mkl = va_arg (ap, int) ;
                va_end (ap) ;
                GB_Global_use_mkl_set (use_mkl != 0) ;
            }
            break ;

        //----------------------------------------------------------------------
        // invalid option
        //----------------------------------------------------------------------

        default : 

            return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,
                    "invalid option field [%d], must be one of:\n"
                    "GxB_HYPER [%d], GxB_FORMAT [%d], GxB_NTHREADS [%d]\n"
                    "GxB_CHUNK [%d], GxB_BURBLE [%d], GxB_GPU_CONTROL [%d]\n"
                    "GxB_GPU_CHUNK [%d], or GxB_MKL [%d]\n",
                    (int) field, (int) GxB_HYPER, (int) GxB_FORMAT,
                    (int) GxB_NTHREADS, (int) GxB_CHUNK, (int) GxB_BURBLE,
                    (int) GxB_GPU_CONTROL, (int) GxB_GPU_CHUNK, (int)
                    GxB_MKL))) ;

    }

    return (GrB_SUCCESS) ;
}

