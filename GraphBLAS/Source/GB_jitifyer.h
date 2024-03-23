//------------------------------------------------------------------------------
// GB_jitifyer.h: definitions for the CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JITIFYER_H
#define GB_JITIFYER_H

#include "GB_jit_kernel_proto.h"

//------------------------------------------------------------------------------
// get list of PreJIT kernels: function pointers and names
//------------------------------------------------------------------------------

void GB_prejit
(
    int32_t *nkernels,      // return # of kernels
    void ***Kernel_handle,  // return list of function pointers to kernels
    void ***Query_handle,   // return list of function pointers to queries
    char ***Name_handle     // return list of kernel names
) ;

//------------------------------------------------------------------------------
// list of jitifyed kernel families
//------------------------------------------------------------------------------

typedef enum
{
    GB_jit_reduce_family    = 1,    // kcode 1
    GB_jit_mxm_family       = 2,    // kcodes 2 to 9
    GB_jit_ewise_family     = 3,    // kcodes 10 to 24
    GB_jit_apply_family     = 4,    // kcodes 25 to 33
    GB_jit_build_family     = 5,    // kcode 34
    GB_jit_select_family    = 6,    // kcodes 35 to 37
    GB_jit_user_op_family   = 7,    // kcode 38
    GB_jit_user_type_family = 8,    // kcode 39
    GB_jit_assign_family    = 9,    // kcodes 40 to 44 (future:: to 78)
}
GB_jit_family ;

//------------------------------------------------------------------------------
// list of jitifyed kernels
//------------------------------------------------------------------------------

typedef enum
{
    // no JIT kernel
    GB_JIT_KERNEL_NONE          = 0,

    // reduce to scalar
    GB_JIT_KERNEL_REDUCE        = 1,  // GB_reduce_to_scalar

    // C<M> = A*B, except for row/col scale (which are ewise methods)
    GB_JIT_KERNEL_AXB_DOT2      = 2,  // GB_AxB_dot2
    GB_JIT_KERNEL_AXB_DOT2N     = 3,  // GB_AxB_dot2n
    GB_JIT_KERNEL_AXB_DOT3      = 4,  // GB_AxB_dot3
    GB_JIT_KERNEL_AXB_DOT4      = 5,  // GB_AxB_dot4
    GB_JIT_KERNEL_AXB_SAXBIT    = 6,  // GB_AxB_saxbit
    GB_JIT_KERNEL_AXB_SAXPY3    = 7,  // GB_AxB_saxpy3
    GB_JIT_KERNEL_AXB_SAXPY4    = 8,  // GB_AxB_saxpy4
    GB_JIT_KERNEL_AXB_SAXPY5    = 9,  // GB_AxB_saxpy5

    // ewise methods:
    GB_JIT_KERNEL_COLSCALE      = 10, // GB_colscale
    GB_JIT_KERNEL_ROWSCALE      = 11, // GB_rowscale
    GB_JIT_KERNEL_ADD           = 12, // GB_add_phase2
    GB_JIT_KERNEL_UNION         = 13, // GB_add_phase2
    GB_JIT_KERNEL_EMULT2        = 14, // GB_emult_02
    GB_JIT_KERNEL_EMULT3        = 15, // GB_emult_03
    GB_JIT_KERNEL_EMULT4        = 16, // GB_emult_04
    GB_JIT_KERNEL_EMULT_BITMAP  = 17, // GB_emult_bitmap
    GB_JIT_KERNEL_EMULT8        = 18, // GB_emult_08_phase2
    GB_JIT_KERNEL_EWISEFA       = 19, // GB_ewise_fulla
    GB_JIT_KERNEL_EWISEFN       = 20, // GB_ewise_fulln
    GB_JIT_KERNEL_APPLYBIND1    = 21, // GB_apply_op, bind1st
    GB_JIT_KERNEL_APPLYBIND2    = 22, // GB_apply_op, bind2nd
    GB_JIT_KERNEL_TRANSBIND1    = 23, // GB_transpose_op, bind1st
    GB_JIT_KERNEL_TRANSBIND2    = 24, // GB_transpose_op, bind2nd

    // apply (unary and idxunary op) methods:
    GB_JIT_KERNEL_APPLYUNOP     = 25, // GB_apply_op, GB_cast_array
    GB_JIT_KERNEL_TRANSUNOP     = 26, // GB_transpose_op, GB_transpose_ix
    GB_JIT_KERNEL_CONVERTS2B    = 27, // GB_convert_2sb
    GB_JIT_KERNEL_CONCAT_SPARSE = 28, // GB_concat_sparse
    GB_JIT_KERNEL_CONCAT_FULL   = 29, // GB_concat_full
    GB_JIT_KERNEL_CONCAT_BITMAP = 30, // GB_concat_bitmap
    GB_JIT_KERNEL_SPLIT_SPARSE  = 31, // GB_split_sparse
    GB_JIT_KERNEL_SPLIT_FULL    = 32, // GB_split_full
    GB_JIT_KERNEL_SPLIT_BITMAP  = 33, // GB_split_bitmap

    // build method:
    GB_JIT_KERNEL_BUILD         = 34, // GB_builder

    // select methods:
    GB_JIT_KERNEL_SELECT1       = 35, // GB_select_sparse
    GB_JIT_KERNEL_SELECT2       = 36, // GB_select_sparse
    GB_JIT_KERNEL_SELECT_BITMAP = 37, // GB_select_bitmap

    // user type and op
    GB_JIT_KERNEL_USERTYPE      = 38, // GxB_Type_new
    GB_JIT_KERNEL_USEROP        = 39, // GxB_*Op_new

    // assign/subassign methods:
    GB_JIT_KERNEL_SUBASSIGN_05d = 40, // GB_subassign_05d
    GB_JIT_KERNEL_SUBASSIGN_06d = 41, // GB_subassign_06d
    GB_JIT_KERNEL_SUBASSIGN_22  = 42, // GB_subassign_22
    GB_JIT_KERNEL_SUBASSIGN_23  = 43, // GB_subassign_23
    GB_JIT_KERNEL_SUBASSIGN_25  = 44, // GB_subassign_25

    //--------------------------------------------------------------------------
    // future:: the following kernels have not been implemented yet
    //--------------------------------------------------------------------------

    // future:: assign/subassign methods: continued
    GB_JIT_KERNEL_SUBASSIGN_01  = 45, // GB_subassign_01
    GB_JIT_KERNEL_SUBASSIGN_02  = 46, // GB_subassign_02
    GB_JIT_KERNEL_SUBASSIGN_03  = 47, // GB_subassign_03
    GB_JIT_KERNEL_SUBASSIGN_04  = 48, // GB_subassign_04
    GB_JIT_KERNEL_SUBASSIGN_05  = 49, // GB_subassign_05
    GB_JIT_KERNEL_SUBASSIGN_06n = 50, // GB_subassign_06n
    GB_JIT_KERNEL_SUBASSIGN_06s = 51, // GB_subassign_06s_and_14
    GB_JIT_KERNEL_SUBASSIGN_07  = 52, // GB_subassign_07
    GB_JIT_KERNEL_SUBASSIGN_08n = 53, // GB_subassign_08n
    GB_JIT_KERNEL_SUBASSIGN_08s = 54, // GB_subassign_08s_and_16
    GB_JIT_KERNEL_SUBASSIGN_09  = 55, // GB_subassign_09
    GB_JIT_KERNEL_SUBASSIGN_10  = 56, // GB_subassign_10_and_18
    GB_JIT_KERNEL_SUBASSIGN_11  = 57, // GB_subassign_11
    GB_JIT_KERNEL_SUBASSIGN_12  = 58, // GB_subassign_12_and_20
    GB_JIT_KERNEL_SUBASSIGN_13  = 59, // GB_subassign_13
    GB_JIT_KERNEL_SUBASSIGN_15  = 60, // GB_subassign_15
    GB_JIT_KERNEL_SUBASSIGN_17  = 61, // GB_subassign_17
    GB_JIT_KERNEL_SUBASSIGN_19  = 62, // GB_subassign_19

    // future:: bitmap assign/subassign:
    GB_JIT_KERNEL_ASSIGN_BITMAP_M_ACC           = 63, // GB_bitmap_assign_M_accum
    GB_JIT_KERNEL_ASSIGN_BITMAP_M_ACC_WHOLE     = 64, // GB_bitmap_assign_M_accum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_M_NOACC         = 65, // GB_bitmap_assign_M_noaccum
    GB_JIT_KERNEL_ASSIGN_BITMAP_M_NOACC_WHOLE   = 66, // GB_bitmap_assign_M_noaccum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_FM_ACC          = 67, // GB_bitmap_assign_fullM_accum
    GB_JIT_KERNEL_ASSIGN_BITMAP_FM_ACC_WHOLE    = 68, // GB_bitmap_assign_fullM_accum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_FM_NOACC        = 69, // GB_bitmap_assign_fullM_noaccum
    GB_JIT_KERNEL_ASSIGN_BITMAP_FM_NOACC_WHOLE  = 70, // GB_bitmap_assign_fullM_noaccum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_ACC         = 71, // GB_bitmap_assign_noM_accum
    GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_ACC_WHOLE   = 72, // GB_bitmap_assign_noM_accum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_NOACC       = 73, // GB_bitmap_assign_noM_noaccum
    GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_NOACC_WHOLE = 74, // GB_bitmap_assign_noM_noaccum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_NM_ACC          = 75, // GB_bitmap_assign_notM_accum
    GB_JIT_KERNEL_ASSIGN_BITMAP_NM_ACC_WHOLE    = 76, // GB_bitmap_assign_notM_accum_whole
    GB_JIT_KERNEL_ASSIGN_BITMAP_NM_NOACC        = 77, // GB_bitmap_assign_notM_noaccum
    GB_JIT_KERNEL_ASSIGN_BITMAP_NM_NOACC_WHOLE  = 78, // GB_bitmap_assign_notM_noaccum_whole

    // future:: subref methods:
    GB_JIT_KERNEL_SUBREF        = 79, // GB_bitmap_subref
    GB_JIT_KERNEL_SUBREF_PHASE3 = 80, // GB_subref_phase3

    // future:: masker methods:
    GB_JIT_KERNEL_MASKER_PHASE1 = 81, // GB_masker_phase1
    GB_JIT_KERNEL_MASKER_PHASE2 = 82, // GB_masker_phase2

    // future:: Kronecker:
    GB_JIT_KERNEL_KRONER        = 83, // GB_kroner

    // future:: utilities:
    GB_JIT_KERNEL_CHECKISO      = 84, // GB_check_if_iso
    GB_JIT_KERNEL_CONVERTBITMAP = 85, // GB_convert_bitmap_worker
    GB_JIT_KERNEL_EXPANDISO     = 86, // GB_expand_iso
    GB_JIT_KERNEL_SORT          = 87, // GB_sort

    //--------------------------------------------------------------------------
    // future:: CUDA kernels
    //--------------------------------------------------------------------------

    GB_JIT_CUDA_KERNEL          = 1000, // no CUDA kernel

    // reduce to scalar in CUDA
    GB_JIT_CUDA_KERNEL_REDUCE   = 1001, // GB_cuda_reduce_to_scalar

    // C<M> = A*B, except for row/col scale (which are ewise methods)
    // ...
    GB_JIT_CUDA_KERNEL_AXB_DOT3 = 1004, // GB_cuda_AxB_dot3

    // ewise methods:
    // ...
    GB_JIT_CUDA_KERNEL_ROWSCALE = 1011,
    // ...

}
GB_jit_kcode ;

//------------------------------------------------------------------------------
// GB_jitifyer_entry: an entry in the jitifyer hash table
//------------------------------------------------------------------------------

// kcode can be reduced to uint8_t, and suffix_len to uint16_t.
// To save space, prejit_index could then be reduced to uint32_t and saved
// in the encoding struct.

struct GB_jit_encoding_struct
{
    uint64_t code ;         // from GB_enumify_*
    uint32_t kcode ;        // which kernel (a GB_jit_kcode)
    uint32_t suffix_len ;   // length of the suffix (0 for builtin)
} ;

typedef struct GB_jit_encoding_struct GB_jit_encoding ;

// prejit_index could be int32_t, but making it int64_t rounds up the size of
// the GB_jit_entry_struct to a multiple of 8 (56 bytes):

struct GB_jit_entry_struct
{
    uint64_t hash ;             // hash code for the problem
    GB_jit_encoding encoding ;  // encoding of the problem, except for suffix
    char *suffix ;              // kernel suffix for user-defined op / types,
                                // NULL for built-in kernels
    void *dl_handle ;           // handle from dlopen, to be passed to dlclose
    void *dl_function ;         // address of kernel function
    int64_t prejit_index ;      // -1: JIT kernel or checked PreJIT kernel
                                // >= 0: index of unchecked PreJIT kernel.
} ;

typedef struct GB_jit_entry_struct GB_jit_entry ;

//------------------------------------------------------------------------------
// GB_jitifyer methods for GraphBLAS
//------------------------------------------------------------------------------

char *GB_jitifyer_libfolder (void) ;    // return path to library folder

GrB_Info GB_jitifyer_load
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

GrB_Info GB_jitifyer_worker
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

GrB_Info GB_jitifyer_load_worker
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    char *kernel_name,          // kernel file name (excluding the path)
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GB_Operator op1,
    GB_Operator op2,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash_encoding (encoding) ;
    GB_jit_encoding *encoding,
    const char *suffix,
    // output
    int64_t *k1,            // location of kernel in PreJIT table
    int64_t *kk             // location of hash entry in hash table
) ;

void GB_jitifyer_entry_free (GB_jit_entry *e) ;

bool GB_jitifyer_insert         // return true if successful, false if failure
(
    // input:
    uint64_t hash,              // hash for the problem
    GB_jit_encoding *encoding,  // primary encoding
    const char *suffix,         // suffix for user-defined types/operators
    void *dl_handle,            // library handle from dlopen (NULL for PreJIT)
    void *dl_function,          // function handle from dlsym
    int32_t prejit_index        // index into PreJIT table; -1 if JIT kernel
) ;

uint64_t GB_jitifyer_hash_encoding
(
    GB_jit_encoding *encoding
) ;

uint64_t GB_jitifyer_hash
(
    const void *bytes,      // any string of bytes
    size_t nbytes,          // # of bytes to hash
    bool jitable            // true if the object can be JIT'd
) ;

// to query a library for its type and operator definitions
typedef GB_JIT_QUERY_PROTO ((*GB_jit_query_func)) ;

bool GB_jitifyer_query
(
    GB_jit_query_func dl_query,
    uint64_t hash,              // hash code for the kernel
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

void GB_jitifyer_cmake_compile (char *kernel_name, uint64_t hash) ;
void GB_jitifyer_direct_compile (char *kernel_name, uint32_t bucket) ;
void GB_jitifyer_nvcc_compile (char *kernel_name, uint32_t bucket) ;

GrB_Info GB_jitifyer_init (void) ;  // initialize the JIT

GrB_Info GB_jitifyer_establish_paths (GrB_Info error_condition) ;
bool GB_jitifyer_path_256 (char *folder) ;

GrB_Info GB_jitifyer_extract_JITpackage (GrB_Info error_condition) ;

void GB_jitifyer_finalize (void) ;              // finalize the JIT
void GB_jitifyer_table_free (bool freeall) ;    // free the JIT table

GrB_Info GB_jitifyer_alloc_space (void) ;

GrB_Info GB_jitifyer_include (void) ;

void GB_jitifyer_set_control (int control) ;
GxB_JIT_Control GB_jitifyer_get_control (void) ;

const char *GB_jitifyer_get_cache_path (void) ;
GrB_Info GB_jitifyer_set_cache_path (const char *new_cache_path) ;
GrB_Info GB_jitifyer_set_cache_path_worker (const char *new_cache_path) ;

const char *GB_jitifyer_get_C_compiler (void) ;
GrB_Info GB_jitifyer_set_C_compiler (const char *new_C_compiler) ;
GrB_Info GB_jitifyer_set_C_compiler_worker (const char *new_C_compiler) ;

const char *GB_jitifyer_get_C_flags (void) ;
GrB_Info GB_jitifyer_set_C_flags (const char *new_C_flags) ;
GrB_Info GB_jitifyer_set_C_flags_worker (const char *new_C_flags) ;

const char *GB_jitifyer_get_C_link_flags (void) ;
GrB_Info GB_jitifyer_set_C_link_flags (const char *new_C_link_flags) ;
GrB_Info GB_jitifyer_set_C_link_flags_worker (const char *new_C_link_flags) ;

const char *GB_jitifyer_get_C_libraries (void) ;
GrB_Info GB_jitifyer_set_C_libraries (const char *new_C_libraries) ;
GrB_Info GB_jitifyer_set_C_libraries_worker (const char *new_C_libraries) ;

const char *GB_jitifyer_get_C_cmake_libs (void) ;
GrB_Info GB_jitifyer_set_C_cmake_libs (const char *new_cmake_libs) ;
GrB_Info GB_jitifyer_set_C_cmake_libs_worker (const char *new_cmake_libs) ;

const char *GB_jitifyer_get_C_preface (void) ;
GrB_Info GB_jitifyer_set_C_preface (const char *new_C_preface) ;
GrB_Info GB_jitifyer_set_C_preface_worker (const char *new_C_preface) ;

const char *GB_jitifyer_get_CUDA_preface (void) ;
GrB_Info GB_jitifyer_set_CUDA_preface (const char *new_CUDA_preface) ;
GrB_Info GB_jitifyer_set_CUDA_preface_worker (const char *new_CUDA_preface) ;

const char *GB_jitifyer_get_error_log (void) ;
GrB_Info GB_jitifyer_set_error_log (const char *new_error_log) ;
GrB_Info GB_jitifyer_set_error_log_worker (const char *new_error_log) ;

bool GB_jitifyer_get_use_cmake (void) ;
void GB_jitifyer_set_use_cmake (bool use_cmake) ;

#endif

