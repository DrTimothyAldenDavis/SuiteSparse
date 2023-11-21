//------------------------------------------------------------------------------
// CHOLMOD/Include/cholmod_template.h: template definitions for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Include/cholmod_internal.h. Copyright (C) 2005-2023,
// Timothy A. Davis.  All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The #including file must #define the following:
//
// Floating-point value, one of the following (default is DOUBLE):
//
//      #define DOUBLE
//      #define SINGLE
//
// pattern/real/complex/zomplex, one of (no default):
//
//      #define PATTERN
//      #define REAL
//      #define COMPLEX
//      #define ZOMPLEX

//------------------------------------------------------------------------------
// undefine current xtype macros, and then define macros for current type
//------------------------------------------------------------------------------

#undef TEMPLATE
#undef TEMPLATE2
#undef TEMPLATE_DTYPE
#undef XTYPE
#undef XTYPE2
#undef XTYPE_OK
#undef ENTRY_IS_NONZERO
#undef ENTRY_IS_ZERO
#undef ENTRY_IS_ONE
#undef IMAG_IS_NONZERO

#undef ABS
#undef ASSEMBLE
#undef ASSIGN
#undef ASSIGN_CONJ
#undef ASSIGN2
#undef ASSIGN2_CONJ
#undef ASSIGN_REAL
#undef ASSIGN_CONJ_OR_NCONJ

#undef MULT
#undef MULTADD
#undef ADD
#undef ADD_REAL
#undef MULTSUB
#undef MULTADDCONJ
#undef MULTSUBCONJ
#undef LLDOT
#undef CLEAR
#undef DIV
#undef DIV_REAL
#undef MULT_REAL
#undef CLEAR_IMAG
#undef LDLDOT
#undef PREFIX

#undef ENTRY_SIZE

#undef XPRINT0
#undef XPRINT1
#undef XPRINT2
#undef XPRINT3

#undef Real

//------------------------------------------------------------------------------
// single/double
//------------------------------------------------------------------------------

#ifdef SINGLE
    // single precision
    #define Real float
    #define TEMPLATE_DTYPE(name) s_ ## name
#else
    // double precision
    #ifndef DOUBLE
    #define DOUBLE
    #endif
    #define Real double
    #define TEMPLATE_DTYPE(name) d_ ## name
#endif

//------------------------------------------------------------------------------
// xtype: pattern/real/complex/zomplex
//------------------------------------------------------------------------------

#ifdef PATTERN

    //--------------------------------------------------------------------------
    // pattern: both double and single
    //--------------------------------------------------------------------------

    #ifdef SINGLE
        #define PREFIX                      ps_
        #define TEMPLATE(name)              PS_TEMPLATE(name)
        #define TEMPLATE2(name)             PS_TEMPLATE(name)
        #define TEMPLATE3(name)             PS_TEMPLATE(name)
    #else
        #define PREFIX                      p_
        #define TEMPLATE(name)              P_TEMPLATE(name)
        #define TEMPLATE2(name)             P_TEMPLATE(name)
    #endif

    #define XTYPE                           CHOLMOD_PATTERN
    #define XTYPE2                          CHOLMOD_REAL
    #define XTYPE_OK(type)                  (TRUE)
    #define ENTRY_IS_NONZERO(ax,az,q)       (TRUE)
    #define ENTRY_IS_ZERO(ax,az,q)          (FALSE)
    #define ENTRY_IS_ONE(ax,az,q)           (TRUE)
    #define IMAG_IS_NONZERO(ax,az,q)        (FALSE)
    #define ENTRY_SIZE                      0

    #define ABS(ax,az,p)                    ((double) 1)
    #define ASSEMBLE(x,z,p,ax,az,q)
    #define ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN_CONJ(x,z,p,ax,az,q)
    #define ASSIGN2(x,z,p,ax,az,q)          P_ASSIGN2(x,z,p,ax,az,q)
    #define ASSIGN2_CONJ(x,z,p,ax,az,q)     P_ASSIGN2(x,z,p,ax,az,q)
    #define ASSIGN_REAL(x,p,ax,q)

    #define MULT(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTADD(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD_REAL(x,p, ax,q, bx,r)
    #define MULTSUB(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define LLDOT(x,p,ax,az,q)
    #define CLEAR(x,z,p)
    #define CLEAR_IMAG(x,z,p)
    #define DIV(x,z,p,ax,az,q)
    #define DIV_REAL(x,z,p, ax,az,q, bx,r)
    #define MULT_REAL(x,z,p, ax,az,q, bx,r)
    #define LDLDOT(x,p, ax,az,q, bx,r)

    #define XPRINT0(x,z,p)                  P_PRINT(0,x,z,p)
    #define XPRINT1(x,z,p)                  P_PRINT(1,x,z,p)
    #define XPRINT2(x,z,p)                  P_PRINT(2,x,z,p)
    #define XPRINT3(x,z,p)                  P_PRINT(3,x,z,p)

#elif defined (REAL)

    //--------------------------------------------------------------------------
    // real: double and single
    //--------------------------------------------------------------------------

    #ifdef SINGLE
        #define PREFIX                      rs_
        #define TEMPLATE(name)              RS_TEMPLATE(name)
        #define TEMPLATE2(name)             RS_TEMPLATE(name)
    #else
        #define PREFIX                      rd_
        #define TEMPLATE(name)              RD_TEMPLATE(name)
        #define TEMPLATE2(name)             RD_TEMPLATE(name)
    #endif

    #define XTYPE                           CHOLMOD_REAL
    #define XTYPE2                          CHOLMOD_REAL
    #define XTYPE_OK(type)                  R_XTYPE_OK(type)
    #define ENTRY_IS_NONZERO(ax,az,q)       R_IS_NONZERO(ax,az,q)
    #define ENTRY_IS_ZERO(ax,az,q)          R_IS_ZERO(ax,az,q)
    #define ENTRY_IS_ONE(ax,az,q)           R_IS_ONE(ax,az,q)
    #define IMAG_IS_NONZERO(ax,az,q)        (FALSE)
    #define ENTRY_SIZE                      1

    #define ABS(ax,az,p)                    R_ABS(ax,az,p)
    #define ASSEMBLE(x,z,p,ax,az,q)         R_ASSEMBLE(x,z,p,ax,az,q)
    #define ASSIGN(x,z,p,ax,az,q)           R_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN_CONJ(x,z,p,ax,az,q)      R_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN2(x,z,p,ax,az,q)          R_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN2_CONJ(x,z,p,ax,az,q)     R_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN_REAL(x,p,ax,q)           R_ASSIGN_REAL(x,p,ax,q)

    #define MULT(x,z,p,ax,az,q,bx,bz,pb)    R_MULT(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTADD(x,z,p,ax,az,q,bx,bz,pb) R_MULTADD(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD(x,z,p,ax,az,q,bx,bz,pb)     R_ADD(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD_REAL(x,p, ax,q, bx,r)       R_ADD_REAL(x,p, ax,q, bx,r)
    #define MULTSUB(x,z,p,ax,az,q,bx,bz,pb) R_MULTSUB(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb) \
        R_MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb) \
        R_MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define LLDOT(x,p,ax,az,q)              R_LLDOT(x,p,ax,az,q)
    #define CLEAR(x,z,p)                    R_CLEAR(x,z,p)
    #define CLEAR_IMAG(x,z,p)               R_CLEAR_IMAG(x,z,p)

    #define DIV(x,z,p,ax,az,q)              R_DIV(x,z,p,ax,az,q)

    #define DIV_REAL(x,z,p, ax,az,q, bx,r)  R_DIV_REAL(x,z,p, ax,az,q, bx,r)
    #define MULT_REAL(x,z,p, ax,az,q, bx,r) R_MULT_REAL(x,z,p, ax,az,q, bx,r)
    #define LDLDOT(x,p, ax,az,q, bx,r)      R_LDLDOT(x,p, ax,az,q, bx,r)

    #define XPRINT0(x,z,p)                  R_PRINT(0,x,z,p)
    #define XPRINT1(x,z,p)                  R_PRINT(1,x,z,p)
    #define XPRINT2(x,z,p)                  R_PRINT(2,x,z,p)
    #define XPRINT3(x,z,p)                  R_PRINT(3,x,z,p)

#elif defined (COMPLEX)

    //--------------------------------------------------------------------------
    // complex: both double and single
    //--------------------------------------------------------------------------

    #ifdef SINGLE
        #define PREFIX                      cs_
        #ifdef NCONJUGATE
            // non-conjugate
            #define TEMPLATE(name)          CS_T_TEMPLATE(name)
            #define TEMPLATE2(name)         CS_T_TEMPLATE(name)
        #else
            // conjugate
            #define TEMPLATE(name)          CS_TEMPLATE(name)
            #define TEMPLATE2(name)         CS_TEMPLATE(name)
        #endif
    #else
        #define PREFIX                      cd_
        #ifdef NCONJUGATE
            // non-conjugate
            #define TEMPLATE(name)          CD_T_TEMPLATE(name)
            #define TEMPLATE2(name)         CD_T_TEMPLATE(name)
        #else
            // conjugate
            #define TEMPLATE(name)          CD_TEMPLATE(name)
            #define TEMPLATE2(name)         CD_TEMPLATE(name)
        #endif
    #endif

    #define ABS(ax,az,p)                    C_ABS(ax,az,p)
    #define ASSEMBLE(x,z,p,ax,az,q)         C_ASSEMBLE(x,z,p,ax,az,q)
    #define ASSIGN(x,z,p,ax,az,q)           C_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN_CONJ(x,z,p,ax,az,q)      C_ASSIGN_CONJ(x,z,p,ax,az,q)
    #define ASSIGN2(x,z,p,ax,az,q)          C_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN2_CONJ(x,z,p,ax,az,q)     C_ASSIGN_CONJ(x,z,p,ax,az,q)
    #define ASSIGN_REAL(x,p,ax,q)           C_ASSIGN_REAL(x,p,ax,q)

    #define XTYPE                           CHOLMOD_COMPLEX
    #define XTYPE2                          CHOLMOD_COMPLEX
    #define XTYPE_OK(type)                  C_XTYPE_OK(type)
    #define ENTRY_IS_NONZERO(ax,az,q)       C_IS_NONZERO(ax,az,q)
    #define ENTRY_IS_ZERO(ax,az,q)          C_IS_ZERO(ax,az,q)
    #define ENTRY_IS_ONE(ax,az,q)           C_IS_ONE(ax,az,q)
    #define IMAG_IS_NONZERO(ax,az,q)        C_IMAG_IS_NONZERO(ax,az,q)
    #define ENTRY_SIZE                      2

    #define MULTADD(x,z,p,ax,az,q,bx,bz,pb) C_MULTADD(x,z,p,ax,az,q,bx,bz,pb)
    #define MULT(x,z,p,ax,az,q,bx,bz,pb)    C_MULT(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD(x,z,p,ax,az,q,bx,bz,pb)     C_ADD(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD_REAL(x,p, ax,q, bx,r)       C_ADD_REAL(x,p, ax,q, bx,r)
    #define MULTSUB(x,z,p,ax,az,q,bx,bz,pb) C_MULTSUB(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb) \
        C_MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb) \
        C_MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define LLDOT(x,p,ax,az,q)              C_LLDOT(x,p,ax,az,q)
    #define CLEAR(x,z,p)                    C_CLEAR(x,z,p)
    #define CLEAR_IMAG(x,z,p)               C_CLEAR_IMAG(x,z,p)

    #ifdef SINGLE
        #define DIV(x,z,p,ax,az,q)          C_S_DIV(x,z,p,ax,az,q)
    #else
        #define DIV(x,z,p,ax,az,q)          C_DIV(x,z,p,ax,az,q)
    #endif

    #define DIV_REAL(x,z,p, ax,az,q, bx,r)  C_DIV_REAL(x,z,p, ax,az,q, bx,r)
    #define MULT_REAL(x,z,p, ax,az,q, bx,r) C_MULT_REAL(x,z,p, ax,az,q, bx,r)
    #define LDLDOT(x,p, ax,az,q, bx,r)      C_LDLDOT(x,p, ax,az,q, bx,r)

    #define XPRINT0(x,z,p)                  C_PRINT(0,x,z,p)
    #define XPRINT1(x,z,p)                  C_PRINT(1,x,z,p)
    #define XPRINT2(x,z,p)                  C_PRINT(2,x,z,p)
    #define XPRINT3(x,z,p)                  C_PRINT(3,x,z,p)

#elif defined (ZOMPLEX)

    //--------------------------------------------------------------------------
    // zomplex: both double and single
    //--------------------------------------------------------------------------

    #ifdef SINGLE
        #define PREFIX                      zs_
        #ifdef NCONJUGATE
            // non-conjugate
            #define TEMPLATE(name)          ZS_T_TEMPLATE(name)
            #define TEMPLATE2(name)         CS_T_TEMPLATE(name)
        #else
            // conjugate
            #define TEMPLATE(name)          ZS_TEMPLATE(name)
            #define TEMPLATE2(name)         CS_TEMPLATE(name)
        #endif
    #else
        #define PREFIX                      zd_
        #ifdef NCONJUGATE
            // non-conjugate
            #define TEMPLATE(name)          ZD_T_TEMPLATE(name)
            #define TEMPLATE2(name)         CD_T_TEMPLATE(name)
        #else
            #define TEMPLATE(name)          ZD_TEMPLATE(name)
            #define TEMPLATE2(name)         CD_TEMPLATE(name)
        #endif
    #endif

    #define ABS(ax,az,p)                    Z_ABS(ax,az,p)
    #define ASSEMBLE(x,z,p,ax,az,q)         Z_ASSEMBLE(x,z,p,ax,az,q)
    #define ASSIGN(x,z,p,ax,az,q)           Z_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN_CONJ(x,z,p,ax,az,q)      Z_ASSIGN_CONJ(x,z,p,ax,az,q)
    #define ASSIGN2(x,z,p,ax,az,q)          Z_ASSIGN(x,z,p,ax,az,q)
    #define ASSIGN2_CONJ(x,z,p,ax,az,q)     Z_ASSIGN_CONJ(x,z,p,ax,az,q)
    #define ASSIGN_REAL(x,p,ax,q)           Z_ASSIGN_REAL(x,p,ax,q)

    #define XTYPE                           CHOLMOD_ZOMPLEX
    #define XTYPE2                          CHOLMOD_ZOMPLEX
    #define XTYPE_OK(type)                  Z_XTYPE_OK(type)
    #define ENTRY_IS_NONZERO(ax,az,q)       Z_IS_NONZERO(ax,az,q)
    #define ENTRY_IS_ZERO(ax,az,q)          Z_IS_ZERO(ax,az,q)
    #define ENTRY_IS_ONE(ax,az,q)           Z_IS_ONE(ax,az,q)
    #define IMAG_IS_NONZERO(ax,az,q)        Z_IMAG_IS_NONZERO(ax,az,q)
    #define ENTRY_SIZE                      1

    #define MULTADD(x,z,p,ax,az,q,bx,bz,pb) Z_MULTADD(x,z,p,ax,az,q,bx,bz,pb)
    #define MULT(x,z,p,ax,az,q,bx,bz,pb)    Z_MULT(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD(x,z,p,ax,az,q,bx,bz,pb)     Z_ADD(x,z,p,ax,az,q,bx,bz,pb)
    #define ADD_REAL(x,p, ax,q, bx,r)       Z_ADD_REAL(x,p, ax,q, bx,r)
    #define MULTSUB(x,z,p,ax,az,q,bx,bz,pb) Z_MULTSUB(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb) \
        Z_MULTADDCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb) \
        Z_MULTSUBCONJ(x,z,p,ax,az,q,bx,bz,pb)
    #define LLDOT(x,p,ax,az,q)              Z_LLDOT(x,p,ax,az,q)
    #define CLEAR(x,z,p)                    Z_CLEAR(x,z,p)
    #define CLEAR_IMAG(x,z,p)               Z_CLEAR_IMAG(x,z,p)

    #ifdef SINGLE
        #define DIV(x,z,p,ax,az,q)          Z_S_DIV(x,z,p,ax,az,q)
    #else
        #define DIV(x,z,p,ax,az,q)          Z_DIV(x,z,p,ax,az,q)
    #endif

    #define DIV_REAL(x,z,p, ax,az,q, bx,r)  Z_DIV_REAL(x,z,p, ax,az,q, bx,r)
    #define MULT_REAL(x,z,p, ax,az,q, bx,r) Z_MULT_REAL(x,z,p, ax,az,q, bx,r)
    #define LDLDOT(x,p, ax,az,q, bx,r)      Z_LDLDOT(x,p, ax,az,q, bx,r)

    #define XPRINT0(x,z,p)                  Z_PRINT(0,x,z,p)
    #define XPRINT1(x,z,p)                  Z_PRINT(1,x,z,p)
    #define XPRINT2(x,z,p)                  Z_PRINT(2,x,z,p)
    #define XPRINT3(x,z,p)                  Z_PRINT(3,x,z,p)

#endif

//------------------------------------------------------------------------------
// conjugate/non-conjugate assignment (for pattern/real/complex/zomplex)
//------------------------------------------------------------------------------

#ifdef NCONJUGATE
    // non-conjugate assign
    #define ASSIGN_CONJ_OR_NCONJ(x,z,p,ax,az,q) ASSIGN(x,z,p,ax,az,q)
#else
    // conjugate assign
    #define ASSIGN_CONJ_OR_NCONJ(x,z,p,ax,az,q) ASSIGN_CONJ(x,z,p,ax,az,q)
#endif

//------------------------------------------------------------------------------
// macros for finding untested code (for developent only)
//------------------------------------------------------------------------------

// These macros are xtype and/or dtype-type specific versions of the GOTCHA
// macro #defined in cholmod_internal.h.

#if 0

    #undef PI_GOTCHA
    #undef PL_GOTCHA

    #undef PDI_GOTCHA
    #undef RDI_GOTCHA
    #undef CDI_GOTCHA
    #undef ZDI_GOTCHA

    #undef PDL_GOTCHA
    #undef RDL_GOTCHA
    #undef CDL_GOTCHA
    #undef ZDL_GOTCHA

    #undef PSI_GOTCHA
    #undef RSI_GOTCHA
    #undef CSI_GOTCHA
    #undef ZSI_GOTCHA

    #undef PSL_GOTCHA
    #undef RSL_GOTCHA
    #undef CSL_GOTCHA
    #undef ZSL_GOTCHA

    #ifdef DOUBLE

        #ifdef CHOLMOD_INT64

            #if defined ( PATTERN )
                #define PL_GOTCHA GOTCHA
                #define PDL_GOTCHA GOTCHA
            #elif defined ( REAL )
                #define RDL_GOTCHA GOTCHA
            #elif defined ( COMPLEX )
                #define CDL_GOTCHA GOTCHA
            #elif defined ( ZOMPLEX )
                #define ZDL_GOTCHA GOTCHA
            #endif

        #else

            #if defined ( PATTERN )
                #define PI_GOTCHA GOTCHA
                #define PDI_GOTCHA GOTCHA
            #elif defined ( REAL )
                #define RDI_GOTCHA GOTCHA
            #elif defined ( COMPLEX )
                #define CDI_GOTCHA GOTCHA
            #elif defined ( ZOMPLEX )
                #define ZDI_GOTCHA GOTCHA
            #endif

        #endif

    #else

        #ifdef CHOLMOD_INT64

            #if defined ( PATTERN )
                #define PL_GOTCHA GOTCHA
                #define PSL_GOTCHA GOTCHA
            #elif defined ( REAL )
                #define RSL_GOTCHA GOTCHA
            #elif defined ( COMPLEX )
                #define CSL_GOTCHA GOTCHA
            #elif defined ( ZOMPLEX )
                #define ZSL_GOTCHA GOTCHA
            #endif

        #else

            #if defined ( PATTERN )
                #define PI_GOTCHA GOTCHA
                #define PSI_GOTCHA GOTCHA
            #elif defined ( REAL )
                #define RSI_GOTCHA GOTCHA
            #elif defined ( COMPLEX )
                #define CSI_GOTCHA GOTCHA
            #elif defined ( ZOMPLEX )
                #define ZSI_GOTCHA GOTCHA
            #endif

        #endif
    #endif

    #ifndef PI_GOTCHA
    #define PI_GOTCHA ;
    #endif

    #ifndef PL_GOTCHA
    #define PL_GOTCHA ;
    #endif

    #ifndef PDI_GOTCHA
    #define PDI_GOTCHA ;
    #endif
    #ifndef RDI_GOTCHA
    #define RDI_GOTCHA ;
    #endif
    #ifndef CDI_GOTCHA
    #define CDI_GOTCHA ;
    #endif
    #ifndef ZDI_GOTCHA
    #define ZDI_GOTCHA ;
    #endif

    #ifndef PDL_GOTCHA
    #define PDL_GOTCHA ;
    #endif
    #ifndef RDL_GOTCHA
    #define RDL_GOTCHA ;
    #endif
    #ifndef CDL_GOTCHA
    #define CDL_GOTCHA ;
    #endif
    #ifndef ZDL_GOTCHA
    #define ZDL_GOTCHA ;
    #endif

    #ifndef PSI_GOTCHA
    #define PSI_GOTCHA ;
    #endif
    #ifndef RSI_GOTCHA
    #define RSI_GOTCHA ;
    #endif
    #ifndef CSI_GOTCHA
    #define CSI_GOTCHA ;
    #endif
    #ifndef ZSI_GOTCHA
    #define ZSI_GOTCHA ;
    #endif

    #ifndef PSL_GOTCHA
    #define PSL_GOTCHA ;
    #endif
    #ifndef RSL_GOTCHA
    #define RSL_GOTCHA ;
    #endif
    #ifndef CSL_GOTCHA
    #define CSL_GOTCHA ;
    #endif
    #ifndef ZSL_GOTCHA
    #define ZSL_GOTCHA ;
    #endif

#endif

