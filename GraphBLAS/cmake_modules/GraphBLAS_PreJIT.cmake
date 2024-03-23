#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS_PreJIT.cmake:  configure the PreJIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0.

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# create a list of files of CPU PreJIT kernels
#-------------------------------------------------------------------------------

file ( GLOB PRE1 "PreJIT/GB_jit_*.c" )
set ( PREJIT "" )
set ( PREPRO "" )
set ( PREQUERY "" )
set ( PREQ "" )
foreach ( PSRC ${PRE1} )
    get_filename_component ( F ${PSRC} NAME_WE )
    list ( APPEND PREJIT ${F} )
    list ( APPEND PREQUERY "JIT_Q (" ${F} "_query)\n" )
    list ( APPEND PREQ "${F}_query" )
    if ( ${F} MATCHES "^GB_jit__add_" )
        list ( APPEND PREPRO "JIT_ADD  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__apply_bind1" )
        list ( APPEND PREPRO "JIT_AP1  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__apply_bind2" )
        list ( APPEND PREPRO "JIT_AP2  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__apply_unop" )
        list ( APPEND PREPRO "JIT_AP0  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot2_" )
        list ( APPEND PREPRO "JIT_DOT2 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot2n_" )
        list ( APPEND PREPRO "JIT_DO2N (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot3_" )
        list ( APPEND PREPRO "JIT_DOT3 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_dot4_" )
        list ( APPEND PREPRO "JIT_DOT4 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxbit" )
        list ( APPEND PREPRO "JIT_SAXB (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxpy3" )
        list ( APPEND PREPRO "JIT_SAX3 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxpy4" )
        list ( APPEND PREPRO "JIT_SAX4 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__AxB_saxpy5" )
        list ( APPEND PREPRO "JIT_SAX5 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__build" )
        list ( APPEND PREPRO "JIT_BLD  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__colscale" )
        list ( APPEND PREPRO "JIT_COLS (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__concat_bitmap" )
        list ( APPEND PREPRO "JIT_CONB (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__concat_full" )
        list ( APPEND PREPRO "JIT_CONF (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__concat_sparse" )
        list ( APPEND PREPRO "JIT_CONS (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__convert_s2b" )
        list ( APPEND PREPRO "JIT_CS2B (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_02" )
        list ( APPEND PREPRO "JIT_EM2  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_03" )
        list ( APPEND PREPRO "JIT_EM3  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_04" )
        list ( APPEND PREPRO "JIT_EM4  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_08" )
        list ( APPEND PREPRO "JIT_EM8  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__emult_bitmap" )
        list ( APPEND PREPRO "JIT_EMB  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__ewise_fulla" )
        list ( APPEND PREPRO "JIT_EWFA (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__ewise_fulln" )
        list ( APPEND PREPRO "JIT_EWFN (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__reduce" )
        list ( APPEND PREPRO "JIT_RED  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__rowscale" )
        list ( APPEND PREPRO "JIT_ROWS (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__select_bitmap" )
        list ( APPEND PREPRO "JIT_SELB (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__select_phase1" )
        list ( APPEND PREPRO "JIT_SEL1 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__select_phase2" )
        list ( APPEND PREPRO "JIT_SEL2 (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__split_bitmap" )
        list ( APPEND PREPRO "JIT_SPB  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__split_full" )
        list ( APPEND PREPRO "JIT_SPF  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__split_sparse" )
        list ( APPEND PREPRO "JIT_SPS  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__subassign" )
        list ( APPEND PREPRO "JIT_SUB  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__trans_bind1" )
        list ( APPEND PREPRO "JIT_TR1  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__trans_bind2" )
        list ( APPEND PREPRO "JIT_TR2  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__trans_unop" )
        list ( APPEND PREPRO "JIT_TR0  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__union" )
        list ( APPEND PREPRO "JIT_UNI  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__user_op" )
        list ( APPEND PREPRO "JIT_UOP  (" ${F} ")\n" )
    elseif ( ${F} MATCHES "^GB_jit__user_type" )
        list ( APPEND PREPRO "JIT_UTYP (" ${F} ")\n" )
    endif ( )
endforeach ( )

list ( JOIN PREPRO "" PREJIT_PROTO )
list ( JOIN PREQUERY "" PREJIT_QUERY )
list ( JOIN PREJIT "\",\n\"" PRENAMES )
list ( LENGTH PREJIT GB_PREJIT_LEN )
list ( JOIN PREJIT ",\n" PREFUNCS )
list ( JOIN PREQ ",\n" PREQFUNCS )

configure_file ( "Config/GB_prejit.c.in"
    "${PROJECT_SOURCE_DIR}/Config/GB_prejit.c"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# create a list of files of CUDA PreJIT kernels
#-------------------------------------------------------------------------------

# FIXME: add CUDA PreJIT kernels.  For example:

#   ...
#   elseif ( ${F} MATCHES "^GB_jit__cuda_reduce" )
#       list ( APPEND PREPRO "JIT_CUDA_RED (" ${F} ")\n" )
#   endif ( )

# configure_file ( "CUDA/Config/GB_prejit.c.in"
#     "${PROJECT_SOURCE_DIR}/CUDA/Config/GB_prejit.c"
#     NEWLINE_STYLE LF )

