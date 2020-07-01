//------------------------------------------------------------------------------
// GB_printf.h: definitions for printing from GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_PRINTF_H
#define GB_PRINTF_H

//------------------------------------------------------------------------------
// global printf and flush function pointers
//------------------------------------------------------------------------------

GB_PUBLIC int (* GB_printf_function ) (const char *format, ...) ;
GB_PUBLIC int (* GB_flush_function  ) ( void ) ;

//------------------------------------------------------------------------------
// printing control
//------------------------------------------------------------------------------

// format strings, normally %llu and %lld, for GrB_Index values
#define GBu "%" PRIu64
#define GBd "%" PRId64

// print to the standard output, and flush the result.  This function can
// print to the MATLAB command window.  No error check is done.  This function
// is meant only for debugging.
#define GBDUMP(...)                             \
{                                               \
    if (GB_printf_function != NULL)             \
    {                                           \
        GB_printf_function (__VA_ARGS__) ;      \
        if (GB_flush_function != NULL)          \
        {                                       \
            GB_flush_function ( ) ;             \
        }                                       \
    }                                           \
    else                                        \
    {                                           \
        printf (__VA_ARGS__) ;                  \
        fflush (stdout) ;                       \
    }                                           \
}

// print to a file f, or to stdout if f is NULL, and check the result.  This
// macro is used by all user-callable GxB_*print and GB_*check functions.
#define GBPR(...)                                                           \
{                                                                           \
    int printf_result = 0 ;                                                 \
    if (f == NULL)                                                          \
    {                                                                       \
        if (GB_printf_function != NULL)                                     \
        {                                                                   \
            printf_result = GB_printf_function (__VA_ARGS__) ;              \
        }                                                                   \
        else                                                                \
        {                                                                   \
            printf_result = printf (__VA_ARGS__) ;                          \
        }                                                                   \
        if (GB_flush_function != NULL)                                      \
        {                                                                   \
            GB_flush_function ( ) ;                                         \
        }                                                                   \
        else                                                                \
        {                                                                   \
            fflush (stdout) ;                                               \
        }                                                                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        printf_result = fprintf (f, __VA_ARGS__)  ;                         \
        fflush (f) ;                                                        \
    }                                                                       \
    if (printf_result < 0)                                                  \
    {                                                                       \
        int err = errno ;                                                   \
        return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,                       \
            "File output error (%d): %s", err, strerror (err)))) ;          \
    }                                                                       \
}

// print if the print level is greater than zero
#define GBPR0(...)                  \
{                                   \
    if (pr != GxB_SILENT)           \
    {                               \
        GBPR (__VA_ARGS__) ;        \
    }                               \
}

// check object->magic and print an error if invalid
#define GB_CHECK_MAGIC(object,kind)                                     \
{                                                                       \
    switch (object->magic)                                              \
    {                                                                   \
        case GB_MAGIC :                                                 \
            /* the object is valid */                                   \
            break ;                                                     \
                                                                        \
        case GB_FREED :                                                 \
            /* dangling pointer! */                                     \
            GBPR0 ("already freed!\n") ;                                \
            return (GB_ERROR (GrB_UNINITIALIZED_OBJECT, (GB_LOG,        \
                "%s is freed: [%s]", kind, name))) ;                    \
                                                                        \
        case GB_MAGIC2 :                                                \
            /* invalid */                                               \
            GBPR0 ("invalid\n") ;                                       \
            return (GB_ERROR (GrB_INVALID_OBJECT, (GB_LOG,              \
                "%s is invalid: [%s]", kind, name))) ;                  \
                                                                        \
        default :                                                       \
            /* uninitialized */                                         \
            GBPR0 ("uninititialized\n") ;                               \
            return (GB_ERROR (GrB_UNINITIALIZED_OBJECT, (GB_LOG,        \
                "%s is uninitialized: [%s]", kind, name))) ;            \
    }                                                                   \
}

//------------------------------------------------------------------------------
// burble
//------------------------------------------------------------------------------

// GB_BURBLE is meant for development use, not production use.  To enable it,
// set GB_BURBLE to 1, either with -DGB_BURBLE=1 as a compiler option, by
// editting the setting above, or by adding the line
//
//      #define GB_BURBLE 1
//
// at the top of any source file, before #including any other file.  After
// enabling it in the library, use GxB_set (GxB_BURBLE, true) to turn it on
// at run time, and GxB_set (GxB_BURBLE, false) to turn it off.  By default,
// the feature is not enabled when SuiteSparse:GraphBLAS is compiled, and
// even then, the setting is set to false by GrB_init.

#if GB_BURBLE

// define the function to use to burble
#define GBBURBLE(...)                               \
{                                                   \
    bool burble = GB_Global_burble_get ( ) ;        \
    if (burble)                                     \
    {                                               \
        GBDUMP (__VA_ARGS__) ;                      \
    }                                               \
}

#if defined ( _OPENMP )

// burble with timing
#define GB_BURBLE_START(func)                       \
double t_burble = 0 ;                               \
bool burble = GB_Global_burble_get ( ) ;            \
{                                                   \
    if (burble)                                     \
    {                                               \
        GBBURBLE (" [ " func " ") ;                 \
        t_burble = GB_OPENMP_GET_WTIME ;            \
    }                                               \
}

#define GB_BURBLE_END                               \
{                                                   \
    if (burble)                                     \
    {                                               \
        t_burble = GB_OPENMP_GET_WTIME - t_burble ; \
        GBBURBLE ("%.3g sec ]\n", t_burble) ;       \
    }                                               \
}

#else

// burble with no timing
#define GB_BURBLE_START(func)                   \
    GBBURBLE (" [ " func " ")

#define GB_BURBLE_END                           \
    GBBURBLE ("]\n")

#endif

#define GB_BURBLE_N(n,...)                      \
    if (n > 1) GBBURBLE (__VA_ARGS__)

#define GB_BURBLE_MATRIX(A, ...)                \
    if (!(A->vlen <= 1 && A->vdim <= 1)) GBBURBLE (__VA_ARGS__)

#else

// no burble
#define GBBURBLE(...)
#define GB_BURBLE_START(func)
#define GB_BURBLE_END
#define GB_BURBLE_N(n,...)
#define GB_BURBLE_MATRIX(A,...)

#endif
#endif

