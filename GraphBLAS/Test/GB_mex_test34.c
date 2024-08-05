//------------------------------------------------------------------------------
// GB_mex_test34: test GrB_get and GrB_set (descriptor)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test34"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#define DGET(desc,value,field)                                  \
{                                                               \
    OK (GrB_Descriptor_get_INT32 (desc, &i, (GrB_Field) (field))) ;           \
    CHECK (i == value) ;                                        \
    OK (GrB_Scalar_clear (s_int32)) ;                           \
    OK (GrB_Descriptor_get_Scalar (desc, s_int32, (GrB_Field) (field))) ;     \
    int32_t iscalar = -1 ;                                      \
    OK (GrB_Scalar_extractElement_INT32 (&iscalar, s_int32)) ;  \
    CHECK (iscalar == value) ;                                  \
    OK (GrB_Descriptor_get_SIZE (desc, &size, GrB_NAME)) ;      \
    CHECK (size == GxB_MAX_NAME_LEN) ;                          \
    OK (GrB_Descriptor_get_String (desc, name, GrB_NAME)) ;     \
    CHECK (MATCH (name, #desc)) ;                               \
}   

#define DSET(desc,value,field)                                  \
{                                                               \
    OK (GrB_Descriptor_set_INT32 (desc, GrB_DEFAULT, (GrB_Field) (field))) ;  \
    OK (GrB_Descriptor_set_INT32 (desc, value, (GrB_Field) (field))) ;        \
    int32_t i2 ;                                                \
    OK (GrB_Descriptor_get_INT32 (desc, &i2, (GrB_Field) (field))) ;          \
    CHECK (i2 == value) ;                                       \
    OK (GrB_Descriptor_set_INT32 (desc, GrB_DEFAULT, (GrB_Field) (field))) ;  \
    OK (GrB_Scalar_setElement_INT32 (s_int32, value)) ;         \
    OK (GrB_Descriptor_set_Scalar (desc, s_int32, (GrB_Field) (field))) ;     \
    int32_t i3 ;                                                \
    OK (GrB_Descriptor_get_INT32 (desc, &i2, (GrB_Field) (field))) ;          \
    CHECK (i2 == value) ;                                       \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info, expected ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Vector v = NULL ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    GrB_Descriptor desc = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char defn [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // GrB_Descriptor get/set
    //--------------------------------------------------------------------------

#if 0

                   // OUTP         MASK           MASK       INP0      INP1
                   //              structural     complement
                   // ===========  ============== ========== ========  ========

// GrB_NULL        // -            -              -          -         -
GrB_DESC_T1      , // -            -              -          -         GrB_TRAN
GrB_DESC_T0      , // -            -              -          GrB_TRAN  -
GrB_DESC_T0T1    , // -            -              -          GrB_TRAN  GrB_TRAN

GrB_DESC_C       , // -            -              GrB_COMP   -         -
GrB_DESC_CT1     , // -            -              GrB_COMP   -         GrB_TRAN
GrB_DESC_CT0     , // -            -              GrB_COMP   GrB_TRAN  -
GrB_DESC_CT0T1   , // -            -              GrB_COMP   GrB_TRAN  GrB_TRAN

GrB_DESC_S       , // -            GrB_STRUCTURE  -          -         -
GrB_DESC_ST1     , // -            GrB_STRUCTURE  -          -         GrB_TRAN
GrB_DESC_ST0     , // -            GrB_STRUCTURE  -          GrB_TRAN  -
GrB_DESC_ST0T1   , // -            GrB_STRUCTURE  -          GrB_TRAN  GrB_TRAN

GrB_DESC_SC      , // -            GrB_STRUCTURE  GrB_COMP   -         -
GrB_DESC_SCT1    , // -            GrB_STRUCTURE  GrB_COMP   -         GrB_TRAN
GrB_DESC_SCT0    , // -            GrB_STRUCTURE  GrB_COMP   GrB_TRAN  -
GrB_DESC_SCT0T1  , // -            GrB_STRUCTURE  GrB_COMP   GrB_TRAN  GrB_TRAN

GrB_DESC_R       , // GrB_REPLACE  -              -          -         -
GrB_DESC_RT1     , // GrB_REPLACE  -              -          -         GrB_TRAN
GrB_DESC_RT0     , // GrB_REPLACE  -              -          GrB_TRAN  -
GrB_DESC_RT0T1   , // GrB_REPLACE  -              -          GrB_TRAN  GrB_TRAN

GrB_DESC_RC      , // GrB_REPLACE  -              GrB_COMP   -         -
GrB_DESC_RCT1    , // GrB_REPLACE  -              GrB_COMP   -         GrB_TRAN
GrB_DESC_RCT0    , // GrB_REPLACE  -              GrB_COMP   GrB_TRAN  -
GrB_DESC_RCT0T1  , // GrB_REPLACE  -              GrB_COMP   GrB_TRAN  GrB_TRAN

GrB_DESC_RS      , // GrB_REPLACE  GrB_STRUCTURE  -          -         -
GrB_DESC_RST1    , // GrB_REPLACE  GrB_STRUCTURE  -          -         GrB_TRAN
GrB_DESC_RST0    , // GrB_REPLACE  GrB_STRUCTURE  -          GrB_TRAN  -
GrB_DESC_RST0T1  , // GrB_REPLACE  GrB_STRUCTURE  -          GrB_TRAN  GrB_TRAN

GrB_DESC_RSC     , // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   -         -
GrB_DESC_RSCT1   , // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   -         GrB_TRAN
GrB_DESC_RSCT0   , // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   GrB_TRAN  -
GrB_DESC_RSCT0T1 ; // GrB_REPLACE  GrB_STRUCTURE  GrB_COMP   GrB_TRAN  GrB_TRAN

#endif

    DGET (GrB_NULL         , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_T1      , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_T0      , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_T0T1    , GrB_DEFAULT, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_C       , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_CT1     , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_CT0     , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_CT0T1   , GrB_DEFAULT, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_S       , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_ST1     , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_ST0     , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_ST0T1   , GrB_DEFAULT, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_SC      , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_SCT1    , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_SCT0    , GrB_DEFAULT, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_SCT0T1  , GrB_DEFAULT, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_R       , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RT1     , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RT0     , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RT0T1   , GrB_REPLACE, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_RC      , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RCT1    , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RCT0    , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RCT0T1  , GrB_REPLACE, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_RS      , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RST1    , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RST0    , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RST0T1  , GrB_REPLACE, GrB_OUTP_FIELD) ;

    DGET (GrB_DESC_RSC     , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RSCT1   , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RSCT0   , GrB_REPLACE, GrB_OUTP_FIELD) ;
    DGET (GrB_DESC_RSCT0T1 , GrB_REPLACE, GrB_OUTP_FIELD) ;



    DGET (GrB_NULL         , GrB_DEFAULT, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_T1      , GrB_DEFAULT, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_T0      , GrB_DEFAULT, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_T0T1    , GrB_DEFAULT, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_C       , GrB_COMP, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_CT1     , GrB_COMP, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_CT0     , GrB_COMP, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_CT0T1   , GrB_COMP, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_S       , GrB_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_ST1     , GrB_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_ST0     , GrB_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_ST0T1   , GrB_STRUCTURE, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_SC      , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_SCT1    , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_SCT0    , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_SCT0T1  , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_R       , GrB_DEFAULT, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RT1     , GrB_DEFAULT, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RT0     , GrB_DEFAULT, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RT0T1   , GrB_DEFAULT, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_RC      , GrB_COMP, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RCT1    , GrB_COMP, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RCT0    , GrB_COMP, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RCT0T1  , GrB_COMP, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_RS      , GrB_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RST1    , GrB_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RST0    , GrB_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RST0T1  , GrB_STRUCTURE, GrB_MASK_FIELD) ;

    DGET (GrB_DESC_RSC     , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RSCT1   , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RSCT0   , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;
    DGET (GrB_DESC_RSCT0T1 , GrB_COMP_STRUCTURE, GrB_MASK_FIELD) ;



    DGET (GrB_NULL         , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_T1      , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_T0      , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_T0T1    , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_C       , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_CT1     , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_CT0     , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_CT0T1   , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_S       , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_ST1     , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_ST0     , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_ST0T1   , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_SC      , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_SCT1    , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_SCT0    , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_SCT0T1  , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_R       , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RT1     , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RT0     , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RT0T1   , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_RC      , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RCT1    , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RCT0    , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RCT0T1  , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_RS      , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RST1    , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RST0    , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RST0T1  , GrB_TRAN   , GrB_INP0_FIELD) ;

    DGET (GrB_DESC_RSC     , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RSCT1   , GrB_DEFAULT, GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RSCT0   , GrB_TRAN   , GrB_INP0_FIELD) ;
    DGET (GrB_DESC_RSCT0T1 , GrB_TRAN   , GrB_INP0_FIELD) ;


    DGET (GrB_NULL         , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_T1      , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_T0      , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_T0T1    , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_C       , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_CT1     , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_CT0     , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_CT0T1   , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_S       , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_ST1     , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_ST0     , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_ST0T1   , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_SC      , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_SCT1    , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_SCT0    , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_SCT0T1  , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_R       , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RT1     , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RT0     , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RT0T1   , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_RC      , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RCT1    , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RCT0    , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RCT0T1  , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_RS      , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RST1    , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RST0    , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RST0T1  , GrB_TRAN   , GrB_INP1_FIELD) ;

    DGET (GrB_DESC_RSC     , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RSCT1   , GrB_TRAN   , GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RSCT0   , GrB_DEFAULT, GrB_INP1_FIELD) ;
    DGET (GrB_DESC_RSCT0T1 , GrB_TRAN   , GrB_INP1_FIELD) ;


    for (int field = GxB_AxB_METHOD ; field <= GxB_IMPORT ; field++)
    {
        DGET (GrB_NULL         , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_T1      , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_T0      , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_T0T1    , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_C       , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_CT1     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_CT0     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_CT0T1   , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_S       , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_ST1     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_ST0     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_ST0T1   , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_SC      , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_SCT1    , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_SCT0    , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_SCT0T1  , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_R       , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RT1     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RT0     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RT0T1   , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_RC      , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RCT1    , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RCT0    , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RCT0T1  , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_RS      , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RST1    , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RST0    , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RST0T1  , GrB_DEFAULT, field) ;

        DGET (GrB_DESC_RSC     , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RSCT1   , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RSCT0   , GrB_DEFAULT, field) ;
        DGET (GrB_DESC_RSCT0T1 , GrB_DEFAULT, field) ;
    }

    OK (GrB_Descriptor_get_String (NULL, name, GrB_NAME)) ;
    CHECK (MATCH (name, "GrB_NULL")) ;

    //--------------------------------------------------------------------------

    OK (GrB_Descriptor_new (&desc)) ;

    DSET (desc, GrB_REPLACE, GrB_OUTP_FIELD) ;
    DSET (desc, GrB_DEFAULT, GrB_OUTP_FIELD) ;

    DSET (desc, GrB_REPLACE, GrB_OUTP) ;
    DSET (desc, GrB_DEFAULT, GrB_OUTP) ;

    DSET (desc, GrB_COMP            , GrB_MASK_FIELD) ;
    DSET (desc, GrB_STRUCTURE       , GrB_MASK_FIELD) ;
    DSET (desc, GrB_COMP_STRUCTURE  , GrB_MASK_FIELD) ;
    DSET (desc, GrB_DEFAULT         , GrB_MASK_FIELD) ;

    DSET (desc, GrB_COMP            , GrB_MASK) ;
    DSET (desc, GrB_STRUCTURE       , GrB_MASK) ;
    DSET (desc, GrB_COMP_STRUCTURE  , GrB_MASK) ;
    DSET (desc, GrB_DEFAULT         , GrB_MASK) ;

    DSET (desc, GrB_TRAN            , GrB_INP0) ;
    DSET (desc, GrB_DEFAULT         , GrB_INP0) ;

    DSET (desc, GrB_TRAN            , GrB_INP1) ;
    DSET (desc, GrB_DEFAULT         , GrB_INP1) ;

    DSET (desc, GxB_AxB_GUSTAVSON   , GxB_AxB_METHOD) ;
    DSET (desc, GxB_AxB_DOT         , GxB_AxB_METHOD) ;
    DSET (desc, GxB_AxB_HASH        , GxB_AxB_METHOD) ;
    DSET (desc, GxB_AxB_SAXPY       , GxB_AxB_METHOD) ;
    DSET (desc, GrB_DEFAULT         , GxB_AxB_METHOD) ;

    DSET (desc, 1                   , GxB_SORT) ;
    DSET (desc, GrB_DEFAULT         , GxB_SORT) ;

    DSET (desc, 1                   , GxB_COMPRESSION) ;
    DSET (desc, GrB_DEFAULT         , GxB_COMPRESSION) ;

    DSET (desc, GxB_FAST_IMPORT     , GxB_IMPORT) ;
    DSET (desc, GxB_SECURE_IMPORT   , GxB_IMPORT) ;
    DSET (desc, GrB_DEFAULT         , GxB_IMPORT) ;

    OK (GrB_Descriptor_get_String_ (desc, name, GrB_NAME)) ;
    CHECK (MATCH (name, "")) ;
    OK (GrB_Descriptor_set_String_ (desc, "user_name", GrB_NAME)) ;
    OK (GrB_Descriptor_get_String_ (desc, name, GrB_NAME)) ;
    printf ("got name: [%s]\n", name) ;
    CHECK (MATCH (name, "user_name")) ;
    OK (GrB_Descriptor_set_String_ (desc, "", GrB_NAME)) ;
    OK (GrB_Descriptor_get_String_ (desc, name, GrB_NAME)) ;
    printf ("got name: [%s]\n", name) ;
    CHECK (MATCH (name, "")) ;
    METHOD (GrB_Descriptor_set_String_ (desc, "yet another name", GrB_NAME)) ;
    OK (GrB_Descriptor_get_String_ (desc, name, GrB_NAME)) ;
    printf ("got name: [%s]\n", name) ;
    CHECK (MATCH (name, "yet another name")) ;
    OK (GrB_Descriptor_get_SIZE_ (desc, &size, GrB_NAME)) ;
    CHECK (size == strlen (name) + 1) ;

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    printf ("\nerror handling:\n") ;
    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Descriptor_get_VOID_ (GrB_DESC_T1, nothing, GrB_NAME)) ;
    ERR (GrB_Descriptor_set_VOID_ (desc, nothing, 0, 0)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Descriptor_get_INT32_ (GrB_DESC_T1, &i, GrB_NAME)) ;
    ERR (GrB_Descriptor_set_INT32_ (GrB_DESC_T1, GrB_REPLACE, (GrB_Field) GrB_OUTP)) ;
    ERR (GrB_Descriptor_set_INT32 (NULL, GrB_REPLACE, (GrB_Field) GrB_OUTP)) ;
    ERR (GrB_Descriptor_get_SIZE_ (GrB_DESC_T1, &size, (GrB_Field) GrB_OUTP)) ;
    ERR (GrB_Descriptor_set_Scalar_ (GrB_DESC_T1, s_int32, (GrB_Field) GrB_MASK)) ;
    ERR (GrB_Descriptor_set_Scalar (NULL, s_int32, (GrB_Field) GrB_MASK)) ;
    ERR (GrB_Descriptor_set_INT32_ (desc, GrB_DEFAULT, GrB_NAME)) ;
    ERR (GrB_Descriptor_set_String_ (GrB_DESC_T1, "newname", GrB_NAME)) ;

    const char *err ;
    ERR (GrB_Descriptor_set_INT32_ (desc, 999, (GrB_Field) GrB_OUTP)) ;
    OK (GrB_Descriptor_error (&err, desc)) ;
    printf ("error: %s\n\n", err) ;

    ERR (GrB_Descriptor_set_INT32_ (desc, 998, (GrB_Field) GrB_MASK)) ;
    OK (GrB_Descriptor_error (&err, desc)) ;
    printf ("error: %s\n\n", err) ;

    ERR (GrB_Descriptor_set_INT32_ (desc, 997, (GrB_Field) GrB_INP0)) ;
    OK (GrB_Descriptor_error (&err, desc)) ;
    printf ("error: %s\n\n", err) ;

    ERR (GrB_Descriptor_set_INT32_ (desc, 996, (GrB_Field) GrB_INP1)) ;
    OK (GrB_Descriptor_error (&err, desc)) ;
    printf ("error: %s\n\n", err) ;

    ERR (GrB_Descriptor_set_INT32_ (desc, 995, (GrB_Field) GxB_AxB_METHOD)) ;
    OK (GrB_Descriptor_error (&err, desc)) ;
    printf ("error: %s\n\n", err) ;

    expected = GrB_EMPTY_OBJECT ;
    OK (GrB_Scalar_clear (s_int32)) ;
    ERR (GrB_Descriptor_set_Scalar_ (desc, s_int32, (GrB_Field) GrB_MASK)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Descriptor_set_VOID_ (desc, nothing, 0, 0)) ;
    ERR (GrB_Descriptor_get_VOID_ (desc, nothing, 0)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&v) ;
    GrB_free (&s) ;
    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&desc) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test34:  all tests passed\n\n") ;
}

