//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_query: tests for cholmod_query
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

void query_test (void)
{

    #ifdef CHOLMOD_HAS_GPL
    int has_gpl = 1 ;
    #else
    int has_gpl = 0 ;
    #endif
    bool has_gpl_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_GPL) ;
    printf ("Query: GPL    %d %d\n", has_gpl, has_gpl_2) ;
    OK (has_gpl == has_gpl_2) ;

    #ifdef CHOLMOD_HAS_CHECK
    int has_check = 1 ;
    #else
    int has_check = 0 ;
    #endif
    bool has_check_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_CHECK) ;
    printf ("Query: CHECK  %d %d\n", has_check, has_check_2) ;
    OK (has_check == has_check_2) ;

    #ifdef CHOLMOD_HAS_CHOLESKY
    int has_chol = 1 ;
    #else
    int has_chol = 0 ;
    #endif
    bool has_chol_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_CHOLESKY) ;
    printf ("Query: CHOL   %d %d\n", has_chol, has_chol_2) ;
    OK (has_chol == has_chol_2) ;

    #ifdef CHOLMOD_HAS_CAMD
    int has_camd = 1 ;
    #else
    int has_camd = 0 ;
    #endif
    bool has_camd_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_CAMD) ;
    printf ("Query: CAMD   %d %d\n", has_camd, has_camd_2) ;
    OK (has_camd == has_camd_2) ;

    #ifdef CHOLMOD_HAS_PARTITION
    int has_part = 1 ;
    #else
    int has_part = 0 ;
    #endif
    bool has_part_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_PARTITION) ;
    printf ("Query: PART   %d %d\n", has_part, has_part_2) ;
    OK (has_part == has_part_2) ;

    #ifdef CHOLMOD_HAS_MATRIXOPS
    int has_ops = 1 ;
    #else
    int has_ops = 0 ;
    #endif
    bool has_ops_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_MATRIXOPS) ;
    printf ("Query: OPS    %d %d\n", has_ops, has_ops_2) ;
    OK (has_ops == has_ops_2) ;

    #ifdef CHOLMOD_HAS_MODIFY
    int has_mod = 1 ;
    #else
    int has_mod = 0 ;
    #endif
    bool has_mod_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_MODIFY) ;
    printf ("Query: MOD    %d %d\n", has_mod, has_mod_2) ;
    OK (has_mod == has_mod_2) ;

    #ifdef CHOLMOD_HAS_SUPERNODAL
    int has_super = 1 ;
    #else
    int has_super = 0 ;
    #endif
    bool has_super_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_SUPERNODAL) ;
    printf ("Query: SUPER  %d %d\n", has_super, has_super_2) ;
    OK (has_super == has_super_2) ;

    #ifdef CHOLMOD_HAS_CUDA
    int has_cuda = 1 ;
    #else
    int has_cuda = 0 ;
    #endif
    bool has_cuda_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_CUDA) ;
    printf ("Query: CUDA   %d %d\n", has_cuda, has_cuda_2) ;
    OK (has_cuda == has_cuda_2) ;

    #ifdef CHOLMOD_HAS_OPENMP
    int has_omp = 1 ;
    #else
    int has_omp = 0 ;
    #endif
    bool has_omp_2 = CHOLMOD(query) (CHOLMOD_QUERY_HAS_OPENMP) ;
    printf ("Query: OMP    %d %d\n", has_omp, has_omp_2) ;
    OK (has_omp == has_omp_2) ;

    bool has_undefined = CHOLMOD(query) (100) ;
    OK (!has_undefined) ;
}

