//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_suitesparse: tests for SuiteSparse_config
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include <complex.h>

typedef double (*hfunc_t) (double, double) ;
typedef int (*dfunc_t) (double, double, double, double, double *, double *) ;
typedef int (*printf_t) (const char *, ...) ;

//------------------------------------------------------------------------------
// my_printf
//------------------------------------------------------------------------------

int my_printf (const char *, ...) ;

int my_printf (const char *fmt, ...)
{
    return (printf ("my_printf [%s]\n", fmt)) ;
}

//------------------------------------------------------------------------------
// my_hypot
//------------------------------------------------------------------------------

double my_hypot (double x, double z) ;

double my_hypot (double x, double y)
{
    printf ("my_hypot\n") ;
    return (sqrt (x*x + y*y)) ;
}

//------------------------------------------------------------------------------
// hypot_test
//------------------------------------------------------------------------------

static double hypot_test (double x, double y, double maxerr)
{
    hfunc_t hfunc = SuiteSparse_config_hypot_func_get ( ) ;
    OKP (hfunc) ;

    double z1 = hfunc (x, y) ;
    double z2 = hypot (x, y) ;
    double err = fabs (z1-z2) ;
    MAXERR (maxerr, err, 1) ;

    double z3 = SuiteSparse_hypot (x, y) ;
    err = fabs (z1-z3) ;
    MAXERR (maxerr, err, 1) ;

    double complex a = CMPLX (x, y) ;
    double z4 = cabs (a) ;
    err = fabs (z1-z4) ;
    MAXERR (maxerr, err, 1) ;

    SuiteSparse_config_hypot_func_set (my_hypot) ;
    double z5 = SuiteSparse_config_hypot (x, y) ;
    err = fabs (z1-z5) ;
    MAXERR (maxerr, err, 1) ;

    hfunc_t hfunc2 = SuiteSparse_config_hypot_func_get ( ) ;
    OK (hfunc2 == my_hypot) ;

    SuiteSparse_config_hypot_func_set (hfunc) ;

    printf ("hypot: %g\n", maxerr) ;
    return (maxerr) ;
}

//------------------------------------------------------------------------------
// my_div
//------------------------------------------------------------------------------

int my_div
(
    double xr, double xi,       // real and imaginary parts of x
    double yr, double yi,       // real and imaginary parts of y
    double *zr, double *zi      // real and imaginary parts of z
) ;

int my_div
(
    double xr, double xi,       // real and imaginary parts of x
    double yr, double yi,       // real and imaginary parts of y
    double *zr, double *zi      // real and imaginary parts of z
)
{
    printf ("my_div\n") ;
    double complex z = CMPLX (xr, xi) / CMPLX (yr, yi) ;
    (*zr) = creal (z) ;
    (*zi) = cimag (z) ;
    return (yr == 0 && yi == 0) ;
}

//------------------------------------------------------------------------------
// div_test
//------------------------------------------------------------------------------

static double div_test (double xr, double xi, double yr, double yi,
    double maxerr)
{
    dfunc_t dfunc = SuiteSparse_config_divcomplex_func_get ( ) ;
    OKP (dfunc) ;
    printf ("\n---------------------div_test:\ndfunc %p\n", dfunc) ;
    printf ("SuiteSparse_divcomplex %p\n", SuiteSparse_divcomplex) ;
    OKP (SuiteSparse_divcomplex) ;

    SuiteSparse_config_divcomplex_func_set (NULL) ;
    dfunc = SuiteSparse_config_divcomplex_func_get ( ) ;
    NOP (dfunc) ;

    SuiteSparse_config_divcomplex_func_set (SuiteSparse_divcomplex) ;
    dfunc = SuiteSparse_config_divcomplex_func_get ( ) ;
    OKP (dfunc) ;

    double zr1 = 0 ;
    double zi1 = 0 ;

    double zr2 = 0 ;
    double zi2 = 0 ;

    double zr3 = 0 ;
    double zi3 = 0 ;

    double zr4 = 0 ;
    double zi4 = 0 ;

    double complex x = CMPLX (xr, xi) ;
    double complex y = CMPLX (yr, yi) ;

    bool ok1 = dfunc (xr, xi, yr, yi, &zr1, &zi1) ;
    bool ok2 = SuiteSparse_divcomplex (xr, xi, yr, yi, &zr2, &zi2) ;
    OK (ok1 == ok2) ;
    double err1 = hypot (zr1-zr2, zi1-zi2) ;
    MAXERR (maxerr, err1, 1) ;

    double complex z4 = x / y ;
    double complex t = CMPLX (zr1, zi1) ;
    double err2 = cabs (t-z4) ;
    MAXERR (maxerr, err2, 1) ;

    bool ok3 = SuiteSparse_config_divcomplex (xr, xi, yr, yi, &zr3, &zi3) ;
    OK (ok1 == ok3) ;
    double err3 = hypot (zr1-zr3, zi1-zi3) ;
    MAXERR (maxerr, err3, 1) ;

    SuiteSparse_config_divcomplex_func_set (my_div) ;
    bool ok4 = SuiteSparse_config_divcomplex (xr, xi, yr, yi, &zr4, &zi4) ;
    OK (ok1 == ok4) ;
    double err4 = hypot (zr1-zr4, zi1-zi4) ;
    MAXERR (maxerr, err4, 1) ;

    printf ("dfunc:   (%g,%g)/(%g,%g) = (%g,%g)\n", xr, xi, yr, yi, zr1, zi1) ;
    printf ("suite:   (%g,%g)/(%g,%g) = (%g,%g)\n", xr, xi, yr, yi, zr2, zi2) ;
    printf ("config1: (%g,%g)/(%g,%g) = (%g,%g)\n", xr, xi, yr, yi, zr3, zi3) ;
    printf ("my_div:  (%g,%g)/(%g,%g) = (%g,%g)\n", xr, xi, yr, yi, zr4, zi4) ;
    printf ("clib:    (%g,%g)/(%g,%g) = (%g,%g)\n",
        creal (x), cimag (x), creal (y), cimag (y), creal (z4), cimag (z4)) ;

    printf ("errs: %g %g %g %g\n", err1, err2, err3, err4) ;

    SuiteSparse_config_divcomplex_func_set (dfunc) ;

    dfunc_t dfunc2 = SuiteSparse_config_divcomplex_func_get ( ) ;
    OK (dfunc == dfunc2) ;

    return (maxerr) ;
}

//------------------------------------------------------------------------------
// suitesparse_tests
//------------------------------------------------------------------------------

double suitesparse_tests (void)
{

    double maxerr = 0 ;

    //--------------------------------------------------------------------------
    // hypot
    //--------------------------------------------------------------------------

    hypot_test (3.4  , 2.3  , maxerr) ;
    hypot_test (2.3  , 3.4  , maxerr) ;
    hypot_test (3.4  , 1e-40, maxerr) ;
    hypot_test (1e-40, 3.4  , maxerr) ;

    //--------------------------------------------------------------------------
    // divcomplex
    //--------------------------------------------------------------------------

    double xr = 1.2 ;
    double xi = -1.2 ;

    double yr = 4.3 ;
    double yi = 3.1 ;

    maxerr = div_test (xr, xi, yr, yi, maxerr) ;
    maxerr = div_test ( 0, xi, yr,  0, maxerr) ;
    maxerr = div_test (xr, xi,  0, yi, maxerr) ;
    maxerr = div_test ( 0, xi,  0, yi, maxerr) ;
    maxerr = div_test (xr,  0,  0, yi, maxerr) ;

    double w = INFINITY ;
    maxerr = div_test (xr, xi, -w,  w, maxerr) ;
    maxerr = div_test (xr, xi,  w, -w, maxerr) ;
    maxerr = div_test (xr, xi, -w, -w, maxerr) ;
    maxerr = div_test (xr, xi,  w,  w, maxerr) ;

    //--------------------------------------------------------------------------
    // BLAS
    //--------------------------------------------------------------------------

    const char *blas_library = SuiteSparse_BLAS_library ( ) ;
    printf ("BLAS library: %s\n", blas_library) ;
    int s = (int) SuiteSparse_BLAS_integer_size ( ) ;
    printf ("BLAS integer size: %d\n", s) ;

    //--------------------------------------------------------------------------
    // printf
    //--------------------------------------------------------------------------

    printf_t pfunc = SuiteSparse_config_printf_func_get ( ) ;
    printf ("printf: %p %p\n", pfunc, printf) ;
    SuiteSparse_config_printf_func_set (NULL) ;
    printf_t pfunc2 = SuiteSparse_config_printf_func_get ( ) ;
    NOP (pfunc2) ;

    SuiteSparse_config_printf_func_set (my_printf) ;
    pfunc2 = SuiteSparse_config_printf_func_get ( ) ;
    OK (pfunc2 == my_printf) ;

    SuiteSparse_config_printf_func_set (pfunc) ;

    //--------------------------------------------------------------------------
    // calloc
    //--------------------------------------------------------------------------

    int *p = SuiteSparse_config_calloc (10, sizeof (int)) ;
    OKP (p) ;
    for (int k = 0 ; k < 10 ; k++)
    {
        OK (p [k] == 0) ;
    }
    SuiteSparse_config_free (p) ;

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    printf ("suitesparse maxerr %g\n", maxerr) ;
    return (maxerr) ;
}

