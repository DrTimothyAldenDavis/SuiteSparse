#===============================================================================
# CHOLMOD/Tcov/Makefile
#===============================================================================

# ------------------------------------------------------------------------------
# CHOLMOD/Tcov Module.  Copyright (C) 2005-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+
# ------------------------------------------------------------------------------

# To compile and run: use "make".
# This test only works in Linux.

default: go

################################################################################

    # SuiteSparse root directory
    SUITESPARSE ?= $(realpath $(CURDIR)/../..)

    BLAS = -lblas
    LAPACK = -llapack

    #---------------------------------------------------------------------------
    # NVIDIA CUDA configuration for CHOLMOD and SPQR
    #---------------------------------------------------------------------------

#   ifneq ($(CUDA),no)
#       CUDA_PATH = $(shell which nvcc 2>/dev/null | sed "s/\/bin\/nvcc//")
#   else
        CUDA_PATH =
#   endif

    ifeq ($(wildcard $(CUDA_PATH)),)
        # CUDA is not present
        CUDA_PATH     =
        GPU_BLAS_PATH =
        GPU_CONFIG    =
        CUDART_LIB    =
        CUBLAS_LIB    =
        CUDA_INC_PATH =
        CUDA_INC      =
        NVCC          = echo
        NVCCFLAGS     =
    else
        # with CUDA for CHOLMOD and SPQR
        GPU_BLAS_PATH = $(CUDA_PATH)
        # GPU_CONFIG must include -DSUITESPARSE_CUDA to compile SuiteSparse for the
        # GPU.  You can add additional GPU-related flags to it as well.
        # with 4 cores (default):
        GPU_CONFIG    = -DSUITESPARSE_CUDA
        # For example, to compile CHOLMOD for 10 CPU cores when using the GPU:
        # GPU_CONFIG  = -DSUITESPARSE_CUDA
        CUDART_LIB    = $(CUDA_PATH)/lib64/libcudart.so
        CUBLAS_LIB    = $(CUDA_PATH)/lib64/libcublas.so
        CUDA_INC_PATH = $(CUDA_PATH)/include/
        CUDA_INC      = -I$(CUDA_INC_PATH)
        NVCC          = $(CUDA_PATH)/bin/nvcc
        NVCCFLAGS     = -Xcompiler -fPIC -O3 \
                            -gencode=arch=compute_53,code=sm_53 \
                            -gencode=arch=compute_60,code=compute_60
    endif

    CHOLMOD_CONFIG ?= $(GPU_CONFIG)

    # remove object files, but keep compiled libraries via 'make clean'
    CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d \
        *.gcda *.gcno *.aux *.bbl *.blg *.log *.toc *.dvi *.lof *.lot

    # also remove compiled libraries, via 'make distclean'
    PURGE = *.so* *.a *.dll *.dylib *.dSYM

    # location of TCOV test output
    TCOV_TMP ?= ../build

################################################################################

# without valgrind
V =
COVER = ./covall
CF = -O0 -g --coverage -fprofile-abs-path -DTEST_COVERAGE -DBLAS32 -fopenmp

# with valgrind
# V = valgrind --suppressions=suppress --quiet
# COVER =
# CF = -O0 -g

# Temp directory
T = $(TCOV_TMP)/CHOLMOD_TCOV_TMP

# Tcov requires gcc
CC = gcc

# to test Tcov without METIS, but with CAMD, CCOLAMD, and CSYMAMD:
# C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(NANTESTS) -DNPARTITION

# to test with everthing
C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(NANTESTS)

# no test coverage
CN = $(CC) -O0 -g -fopenmp $(CHOLMOD_CONFIG) $(NANTESTS)

# LDLIBS = -L$(SUITESPARSE)/lib -lsuitesparseconfig \
#   	-lm $(LAPACK) $(BLAS) -lrt -Wl,-rpath=$(SUITESPARSE)/lib

LDLIBS = -lm $(LAPACK) $(BLAS)

#-------------------------------------------------------------------------------
# With the CUDA BLAS:
ifneq ($(GPU_CONFIG),)
LDLIBS += $(CUDART_LIB) $(CUBLAS_LIB)
endif
#-------------------------------------------------------------------------------

I = -I.. -I../../AMD/Include -I../../COLAMD/Include \
	-I../SuiteSparse_metis/include -I../../CCOLAMD/Include \
	-I../../CAMD/Include \
	-I../Include -I../../SuiteSparse_config $(CUDA_INC) \
        -I../Check -I../Cholesky -I../Demo -I../Supernodal \
        -I../Partition -I../Modify -I../MatrixOps -I../GPU \
        -I../SuiteSparse_metis \
        -I../SuiteSparse_metis/GKlib \
        -I../SuiteSparse_metis/libmetis \
        -I../../AMD/Source -I../../CAMD/Source \
        -I../../COLAMD/Source -I../../CCOLAMD/Source \
        -I../Utility

I += $(GPU_CONFIG)

ccode: cm cl zdemo ldemo cmread clread

TEST = cm.c \
    test_ops.c  null.c null2.c lpdemo.c memory.c solve.c aug.c unpack.c \
    raw_factor.c cctest.c ctest.c amdtest.c camdtest.c huge.c

LTEST = cl.c amdtest_l.c camdtest_l.c huge_l.c

INC =   ../Include/cholmod.h \
	../Include/cholmod_internal.h \
	../Include/cholmod_template.h

AMDSRC = ../../AMD/Source/amd_1.c \
	../../AMD/Source/amd_2.c \
	../../AMD/Source/amd_aat.c \
	../../AMD/Source/amd_control.c \
	../../AMD/Source/amd_defaults.c \
	../../AMD/Source/amd_info.c \
	../../AMD/Source/amd_order.c \
	../../AMD/Source/amd_postorder.c \
	../../AMD/Source/amd_post_tree.c \
	../../AMD/Source/amd_preprocess.c \
	../../AMD/Source/amd_valid.c \
	../../AMD/Include/amd.h \
	../../AMD/Include/amd_internal.h

AMDOBJ = \
	zz_amd_1.o \
	zz_amd_2.o \
	zz_amd_aat.o \
	zz_amd_control.o \
	zz_amd_defaults.o \
	zz_amd_info.o \
	zz_amd_order.o \
	zz_amd_postorder.o \
	zz_amd_post_tree.o \
	zz_amd_preprocess.o \
	zz_amd_valid.o
# add this to AMDOBJ if debugging is enabled
#	zz_amd_dump.o \

LAMDOBJ = \
	zl_amd_1.o \
	zl_amd_2.o \
	zl_amd_aat.o \
	zl_amd_control.o \
	zl_amd_defaults.o \
	zl_amd_info.o \
	zl_amd_order.o \
	zl_amd_postorder.o \
	zl_amd_post_tree.o \
	zl_amd_preprocess.o \
	zl_amd_valid.o
# add this to LAMDOBJ if debugging is enabled
#	zl_amd_dump.o \

COLAMDSRC = ../../COLAMD/Source/colamd.c

COLAMDOBJ = zz_colamd.o

LCOLAMDOBJ = zl_colamd.o

#-------------------------------------------------------------------------------
# When using the Partition Module:

CCOLAMDSRC = \
	../../CCOLAMD/Source/ccolamd.c \
	../../CCOLAMD/Include/ccolamd.h

CCOLAMDOBJ = zz_ccolamd.o

LCCOLAMDOBJ = zl_ccolamd.o

$(CCOLAMDOBJ): $(CCOLAMDSRC)

$(LCCOLAMDOBJ): $(CCOLAMDSRC)

IPARTITION_OBJ = \
	z_ccolamd.o \
	z_csymamd.o \
	z_camd.o \
	z_metis.o \
	n_metis_wrapper.o \
	z_nesdis.o

LPARTITION_OBJ = \
	l_ccolamd.o \
	l_csymamd.o \
	l_camd.o \
	l_metis.o \
	n_metis_wrapper.o \
	l_nesdis.o

CAMDSRC = ../../CAMD/Source/camd_1.c \
	../../CAMD/Source/camd_2.c \
	../../CAMD/Source/camd_aat.c \
	../../CAMD/Source/camd_control.c \
	../../CAMD/Source/camd_defaults.c \
	../../CAMD/Source/camd_info.c \
	../../CAMD/Source/camd_order.c \
	../../CAMD/Source/camd_postorder.c \
	../../CAMD/Source/camd_preprocess.c \
	../../CAMD/Source/camd_valid.c \
	../../CAMD/Include/camd.h \
	../../CAMD/Include/camd_internal.h

CAMDOBJ = \
	zz_camd_1.o \
	zz_camd_2.o \
	zz_camd_aat.o \
	zz_camd_control.o \
	zz_camd_defaults.o \
	zz_camd_info.o \
	zz_camd_order.o \
	zz_camd_postorder.o \
	zz_camd_preprocess.o \
	zz_camd_valid.o
# add this to CAMDOBJ if debugging is enabled
#	zz_camd_dump.o \

LCAMDOBJ = \
	zl_camd_1.o \
	zl_camd_2.o \
	zl_camd_aat.o \
	zl_camd_control.o \
	zl_camd_defaults.o \
	zl_camd_info.o \
	zl_camd_order.o \
	zl_camd_postorder.o \
	zl_camd_preprocess.o \
	zl_camd_valid.o
# add this to LCAMDOBJ if debugging is enabled
#	zl_camd_dump.o \

$(CAMDOBJ): $(CAMDSRC)

$(LCAMDOBJ): $(CAMDSRC)

#-------------------------------------------------------------------------------
# If you compile with -DNPARTITION, you may replace the above definitions
# with empty ones (see immediately below), and then you do not need a copy of
# CCOLAMD:
# CCOLAMDSRC =
# CCOLAMDOBJ = 
# LCCOLAMDOBJ =
# IPARTITION_OBJ = 
# LPARTITION_OBJ = 
# CAMDSRC = 
# CAMDOBJ = 
# LCAMDOBJ = 
#-------------------------------------------------------------------------------

IUTIL_OBJ = \
    ui_change_factor.o \
    ui_transpose_unsym.o \
    ui_transpose_sym.o \
    ui_transpose.o \
    ui_ptranspose.o \
    ui_sort.o \
    ui_aat.o \
    ui_factor_to_sparse.o \
    ui_reallocate_column.o \
    ui_copy_factor.o \
    ui_pack_factor.o \
    ui_reallocate_factor.o \
    ui_free_factor.o \
    ui_alloc_factor.o \
    ui_allocate_factor.o \
    ui_copy.o \
    ui_add.o \
    ui_add_size_t.o \
    ui_alloc_work.o \
    ui_allocate_dense.o \
    ui_allocate_sparse.o \
    ui_allocate_triplet.o \
    ui_allocate_work.o \
    ui_band.o \
    ui_band_nnz.o \
    ui_calloc.o \
    ui_clear_flag.o \
    ui_copy_dense.o \
    ui_copy_dense2.o \
    ui_copy_sparse.o \
    ui_copy_triplet.o \
    ui_cumsum.o \
    ui_dbound.o \
    ui_defaults.o \
    ui_dense_nnz.o \
    ui_dense_to_sparse.o \
    ui_divcomplex.o \
    ui_ensure_dense.o \
    ui_error.o \
    ui_eye.o \
    ui_finish.o \
    ui_free.o \
    ui_free_dense.o \
    ui_free_sparse.o \
    ui_free_triplet.o \
    ui_free_work.o \
    ui_hypot.o \
    ui_malloc.o \
    ui_maxrank.o \
    ui_mult_size_t.o \
    ui_nnz.o \
    ui_ones.o \
    ui_realloc.o \
    ui_realloc_multiple.o \
    ui_reallocate_sparse.o \
    ui_reallocate_triplet.o \
    ui_sbound.o \
    ui_score_comp.o \
    ui_set_empty.o \
    ui_sparse_to_dense.o \
    ui_sparse_to_triplet.o \
    ui_speye.o \
    ui_spzeros.o \
    ui_start.o \
    ui_triplet_to_sparse.o \
    ui_version.o \
    ui_xtype.o \
    ui_zeros.o

LUTIL_OBJ = \
    ul_change_factor.o \
    ul_transpose_unsym.o \
    ul_transpose_sym.o \
    ul_transpose.o \
    ul_ptranspose.o \
    ul_sort.o \
    ul_aat.o \
    ul_factor_to_sparse.o \
    ul_reallocate_column.o \
    ul_copy_factor.o \
    ul_pack_factor.o \
    ul_reallocate_factor.o \
    ul_free_factor.o \
    ul_alloc_factor.o \
    ul_allocate_factor.o \
    ul_copy.o \
    ul_add.o \
    ul_add_size_t.o \
    ul_alloc_work.o \
    ul_allocate_dense.o \
    ul_allocate_sparse.o \
    ul_allocate_triplet.o \
    ul_allocate_work.o \
    ul_band.o \
    ul_band_nnz.o \
    ul_calloc.o \
    ul_clear_flag.o \
    ul_copy_dense.o \
    ul_copy_dense2.o \
    ul_copy_sparse.o \
    ul_copy_triplet.o \
    ul_cumsum.o \
    ul_dbound.o \
    ul_defaults.o \
    ul_dense_nnz.o \
    ul_dense_to_sparse.o \
    ul_divcomplex.o \
    ul_ensure_dense.o \
    ul_error.o \
    ul_eye.o \
    ul_finish.o \
    ul_free.o \
    ul_free_dense.o \
    ul_free_sparse.o \
    ul_free_triplet.o \
    ul_free_work.o \
    ul_hypot.o \
    ul_malloc.o \
    ul_maxrank.o \
    ul_mult_size_t.o \
    ul_nnz.o \
    ul_ones.o \
    ul_realloc.o \
    ul_realloc_multiple.o \
    ul_reallocate_sparse.o \
    ul_reallocate_triplet.o \
    ul_sbound.o \
    ul_score_comp.o \
    ul_set_empty.o \
    ul_sparse_to_dense.o \
    ul_sparse_to_triplet.o \
    ul_speye.o \
    ul_spzeros.o \
    ul_start.o \
    ul_triplet_to_sparse.o \
    ul_version.o \
    ul_xtype.o \
    ul_zeros.o

IOBJ = $(IUTIL_OBJ) \
	z_check.o \
	z_read.o \
	z_write.o \
	z_amd.o \
	z_analyze.o \
	z_colamd.o \
	z_etree.o \
	z_factorize.o \
	z_postorder.o \
	z_rcond.o \
	z_resymbol.o \
	z_rowcolcounts.o \
	z_rowfac.o \
	z_solve.o \
	z_spsolve.o \
	z_drop.o \
	z_horzcat.o \
	z_norm.o \
	z_scale.o \
	z_sdmult.o \
	z_ssmult.o \
	z_submatrix.o \
	z_vertcat.o \
	z_symmetry.o \
	z_rowadd.o \
	z_rowdel.o \
	z_updown.o \
	z_super_numeric.o \
	z_super_solve.o \
	z_super_symbolic.o \
	$(IPARTITION_OBJ)

LOBJ = $(LUTIL_OBJ) \
	l_check.o \
	l_read.o \
	l_write.o \
	l_amd.o \
	l_analyze.o \
	l_colamd.o \
	l_etree.o \
	l_factorize.o \
	l_postorder.o \
	l_rcond.o \
	l_resymbol.o \
	l_rowcolcounts.o \
	l_rowfac.o \
	l_solve.o \
	l_spsolve.o \
	l_drop.o \
	l_horzcat.o \
	l_norm.o \
	l_scale.o \
	l_sdmult.o \
	l_ssmult.o \
	l_submatrix.o \
	l_vertcat.o \
	l_symmetry.o \
	l_rowadd.o \
	l_rowdel.o \
	l_updown.o \
	l_super_numeric.o \
	l_super_solve.o \
	l_super_symbolic.o \
	$(LPARTITION_OBJ)

ifneq ($(GPU_CONFIG),)
    IGPU = l_gpu_kernels.o z_gpu.o
    LGPU = l_gpu_kernels.o l_gpu.o
else
    IGPU = 
    LGPU = 
endif

CONFIG = zz_SuiteSparse_config.o
# CONFIG =

ILOBJ = u_mult_uint64_t.o u_memdebug.o

IALL = $(IOBJ) $(AMDOBJ)  $(COLAMDOBJ)  $(CCOLAMDOBJ)  $(CAMDOBJ)   $(CONFIG) $(ILOBJ) $(IGPU)

LALL = $(LOBJ) $(LAMDOBJ) $(LCOLAMDOBJ) $(LCCOLAMDOBJ) $(LCAMDOBJ)  $(CONFIG) $(ILOBJ) $(LGPU)


cm: $(IALL) $(TEST) cm.h Makefile
	$(C) $(I) $(TEST) -o cm $(IALL) $(LDLIBS)

cl: $(LALL) $(LTEST) cm.h Makefile cm.c \
    test_ops.c  null.c null2.c lpdemo.c memory.c solve.c aug.c unpack.c \
    raw_factor.c cctest.c ctest.c amdtest.c camdtest.c huge.c
	$(C) $(I) $(LTEST) -o cl $(LALL) $(LDLIBS)

cmread: $(IALL) cmread.c Makefile
	$(C) $(I) cmread.c -o cmread $(IALL) $(LDLIBS)

clread: $(LALL) clread.c Makefile
	$(C) $(I) clread.c -o clread $(LALL) $(LDLIBS)

zdemo: $(IALL) ../Demo/cholmod_demo.c cm.h Makefile \
	../Demo/cholmod_demo.h
	- ln -s ../Demo/cholmod_demo.c zdemo.c
	$(C) $(I) -I../Demo zdemo.c -o zdemo $(IALL) $(LDLIBS)

ldemo: $(LALL) ../Demo/cholmod_l_demo.c cm.h Makefile \
	../Demo/cholmod_demo.h
	- ln -s ../Demo/cholmod_l_demo.c ldemo.c
	$(C) $(I) -I../Demo ldemo.c -o ldemo $(LALL) $(LDLIBS)

compile: zdemo ldemo cmread clread cm cl

go: zdemo ldemo cmread clread cm cl
	mkdir -p $(T)
	$(V) ./cl    < Matrix/galenet               > $(T)/l_galenet.out
	$(V) ./cl -m < Matrix/z5lo                  > $(T)/l_z5lo.out
	$(V) ./zdemo   ../Demo/Matrix/bcsstk01.tri  > $(T)/demo_k1.out
	$(V) ./zdemo   ../Demo/Matrix/bcsstk02.tri  > $(T)/demo_k2.out
	$(V) ./zdemo < ../Demo/Matrix/lp_afiro.tri  > $(T)/demo_afiro.out
	$(V) ./zdemo < ../Demo/Matrix/can___24.mtx  > $(T)/demo_can24.out
	$(V) ./zdemo < ../Demo/Matrix/c.tri         > $(T)/demo_c.out
	$(V) ./zdemo < ../Demo/Matrix/d.tri         > $(T)/demo_d.out
	$(V) ./zdemo < ../Demo/Matrix/up.tri        > $(T)/demo_up.out
	$(V) ./zdemo < ../Demo/Matrix/c.mtx         > $(T)/demo_c_mtx.out
	$(V) ./zdemo < ../Demo/Matrix/0.tri         > $(T)/demo_0.out
	$(V) ./zdemo < Matrix/3_2                   > $(T)/demo_3_2.out
	$(V) ./zdemo < Matrix/c5lo                  > $(T)/demo_c5lo.out
	$(V) ./zdemo < Matrix/c10                   > $(T)/demo_c10.out
	$(V) ./zdemo no_such_file                   > $(T)/demo_no_such_file.out
	$(V) ./zdemo ../Demo/Matrix/mangle1.mtx     > $(T)/demo_mangle1.out
	$(V) ./zdemo ../Demo/Matrix/mangle2.mtx     > $(T)/demo_mangle2.out
	$(V) ./zdemo ../Demo/Matrix/mangle3.mtx     > $(T)/demo_mangle3.out
	$(V) ./zdemo ../Demo/Matrix/mangle4.mtx     > $(T)/demo_mangle4.out
	$(V) ./zdemo ../Demo/Matrix/pts5ldd03.mtx   > $(T)/demo_pts5ldd03.out
	$(V) ./ldemo   ../Demo/Matrix/bcsstk01.tri  > $(T)/ldemo_k1.out
	$(V) ./ldemo   ../Demo/Matrix/bcsstk02.tri  > $(T)/ldemo_k2.out
	$(V) ./ldemo < ../Demo/Matrix/lp_afiro.tri  > $(T)/ldemo_afiro.out
	$(V) ./ldemo < ../Demo/Matrix/can___24.mtx  > $(T)/ldemo_can24.out
	$(V) ./ldemo < ../Demo/Matrix/c.tri         > $(T)/ldemo_c.out
	$(V) ./ldemo   ../Demo/Matrix/c.tri 1       > $(T)/ldemo_c_zomplex.out
	$(V) ./ldemo < ../Demo/Matrix/d.tri         > $(T)/ldemo_d.out
	$(V) ./ldemo   ../Demo/Matrix/d.tri 1       > $(T)/ldemo_d.out
	$(V) ./ldemo < ../Demo/Matrix/up.tri        > $(T)/ldemo_up.out
	$(V) ./ldemo   ../Demo/Matrix/up.tri 1      > $(T)/ldemo_up_zomplex.out
	$(V) ./ldemo < ../Demo/Matrix/c.mtx         > $(T)/ldemo_c_mtx.out
	$(V) ./ldemo   ../Demo/Matrix/c.mtx 1       > $(T)/ldemo_c_mtx_zomplex.out
	$(V) ./ldemo < ../Demo/Matrix/0.tri         > $(T)/ldemo_0.out
	$(V) ./ldemo < Matrix/3_2                   > $(T)/ldemo_3_2.out
	$(V) ./ldemo < Matrix/c5lo                  > $(T)/ldemo_c5lo.out
	$(V) ./ldemo < Matrix/c10                   > $(T)/ldemo_c10.out
	$(V) ./ldemo no_such_file                   > $(T)/ldemo_no_such_file.out
	$(V) ./ldemo ../Demo/Matrix/mangle1.mtx     > $(T)/ldemo_mangle1.out
	$(V) ./ldemo ../Demo/Matrix/mangle2.mtx     > $(T)/ldemo_mangle2.out
	$(V) ./ldemo ../Demo/Matrix/mangle3.mtx     > $(T)/ldemo_mangle3.out
	$(V) ./ldemo ../Demo/Matrix/mangle4.mtx     > $(T)/ldemo_mangle4.out
	$(V) ./ldemo ../Demo/Matrix/pts5ldd03.mtx   > $(T)/ldemo_pts5ldd03.out
	- grep resid $(T)/demo*
	$(V) ./cmread no_such_file                  > $(T)/no_such_file.out
	$(V) ./cmread Matrix/crud1                  > $(T)/crud1.out
	$(V) ./cmread Matrix/crud2                  > $(T)/crud2.out
	$(V) ./cmread Matrix/fullcrud.mtx           > $(T)/fullcrud.out
	$(V) ./cmread Matrix/fullcrud1.mtx          > $(T)/fullcrud1.out
	$(V) ./cmread Matrix/fullcrud2.mtx          > $(T)/fullcrud2.out
	$(V) ./cmread Matrix/3by0.mtx               > $(T)/3by0.out
	$(V) ./cmread Matrix/fullrza.mtx            > $(T)/fullrza.out
	$(V) ./cmread Matrix/fullrsa.mtx            > $(T)/fullrsa.out
	$(V) ./cmread Matrix/fullcsa.mtx            > $(T)/fullcsa.out
	$(V) ./cmread Matrix/fullcza.mtx            > $(T)/fullcza.out
	$(V) ./cmread Matrix/fullcha.mtx            > $(T)/fullcha.out
	$(V) ./cmread Matrix/cha.mtx                > $(T)/cha.out
	$(V) ./cmread Matrix/cza.mtx                > $(T)/cza.out
	$(V) ./cmread Matrix/csa.mtx                > $(T)/csa.out
	$(V) ./cmread Matrix/one                    > $(T)/one.out
	$(V) ./cmread Matrix/rza.mtx                > $(T)/rza.out
	$(V) ./cmread ../Demo/Matrix/mangle5.tri    > $(T)/mangle5.out
	$(V) ./cmread ../Demo/Matrix/mangle6.tri    > $(T)/mangle6.out
	$(V) ./cmread ../Demo/Matrix/mangle7.tri    > $(T)/mangle6.out
	$(V) ./cmread ../Demo/Matrix/mangle8.tri    > $(T)/mangle8.out
	$(V) ./cmread ../Demo/Matrix/empty.tri      > $(T)/empty.out
	$(V) ./cmread ../Demo/Matrix/one.tri        > $(T)/one.out
	$(V) ./cmread Matrix/plskz362.mtx           > $(T)/plskz363.out
	$(V) ./cmread Matrix/2diag.tri              > $(T)/2diag.out
	$(V) ./cmread Matrix/r5lo                   > $(T)/r5lo.out
	$(V) ./cmread Matrix/r5lo2                  > $(T)/r5lo2.out
	- diff $(T)/r5lo.out $(T)/r5lo2.out
	$(V) ./cmread Matrix/cs.mtx                 > $(T)/cs.out
	$(V) ./cmread Matrix/2lo.tri                > $(T)/2lo.out
	$(V) ./cmread Matrix/2.tri                  > $(T)/2.out
	$(V) ./cmread Matrix/2up.tri                > $(T)/2up.out
	$(V) ./cmread Matrix/huge.tri               > $(T)/huge.out
	$(V) ./cmread Matrix/1e99                   > $(T)/1e99.out
	$(V) ./clread no_such_file                  > $(T)/l_no_such_file.out
	$(V) ./clread Matrix/crud1                  > $(T)/l_crud1.out
	$(V) ./clread Matrix/crud2                  > $(T)/l_crud2.out
	$(V) ./clread Matrix/fullcrud.mtx           > $(T)/l_fullcrud.out
	$(V) ./clread Matrix/fullcrud1.mtx          > $(T)/l_fullcrud1.out
	$(V) ./clread Matrix/fullcrud2.mtx          > $(T)/l_fullcrud2.out
	$(V) ./clread Matrix/3by0.mtx               > $(T)/l_3by0.out
	$(V) ./clread Matrix/fullrza.mtx            > $(T)/l_fullrza.out
	$(V) ./clread Matrix/fullrsa.mtx            > $(T)/l_fullrsa.out
	$(V) ./clread Matrix/fullcsa.mtx            > $(T)/l_fullcsa.out
	$(V) ./clread Matrix/fullcza.mtx            > $(T)/l_fullcza.out
	$(V) ./clread Matrix/fullcha.mtx            > $(T)/l_fullcha.out
	$(V) ./clread Matrix/cha.mtx                > $(T)/l_cha.out
	$(V) ./clread Matrix/cza.mtx                > $(T)/l_cza.out
	$(V) ./clread Matrix/csa.mtx                > $(T)/l_csa.out
	$(V) ./clread Matrix/one                    > $(T)/l_one.out
	$(V) ./clread Matrix/rza.mtx                > $(T)/l_rza.out
	$(V) ./clread ../Demo/Matrix/mangle5.tri    > $(T)/l_mangle5.out
	$(V) ./clread ../Demo/Matrix/mangle6.tri    > $(T)/l_mangle6.out
	$(V) ./clread ../Demo/Matrix/mangle7.tri    > $(T)/l_mangle6.out
	$(V) ./clread ../Demo/Matrix/mangle8.tri    > $(T)/l_mangle8.out
	$(V) ./clread ../Demo/Matrix/empty.tri      > $(T)/l_empty.out
	$(V) ./clread ../Demo/Matrix/one.tri        > $(T)/l_one.out
	$(V) ./clread Matrix/plskz362.mtx           > $(T)/l_plskz363.out
	$(V) ./clread Matrix/2diag.tri              > $(T)/l_2diag.out
	$(V) ./clread Matrix/r5lo                   > $(T)/l_r5lo.out
	$(V) ./clread Matrix/r5lo2                  > $(T)/l_r5lo2.out
	- diff $(T)/r5lo.out $(T)/r5lo2.out
	$(V) ./clread Matrix/cs.mtx                 > $(T)/l_cs.out
	$(V) ./clread Matrix/2lo.tri                > $(T)/l_l_2lo.out
	$(V) ./clread Matrix/2.tri                  > $(T)/l_2.out
	$(V) ./clread Matrix/2up.tri                > $(T)/l_2up.out
	$(V) ./clread Matrix/huge.tri               > $(T)/l_huge.out
	$(V) ./clread Matrix/1e99                   > $(T)/l_1e99.out
	$(V) ./cm    < Matrix/galenet               > $(T)/galenet.out
	- $(COVER)
	$(V) ./cm    < Matrix/5by50                 > $(T)/5by50.out
	$(V) ./cl    < Matrix/5by50                 > $(T)/l_5by50.out
	- $(COVER)
	$(V) ./cm    < Matrix/r5lo                  > $(T)/r5lo.out 
	$(V) ./cl    < Matrix/r5lo                  > $(T)/l_r5lo.out
	$(V) ./cm    < Matrix/r5up                  > $(T)/r5up.out
	$(V) ./cl    < Matrix/r5up                  > $(T)/l_r5up.out
	$(V) ./cm    < Matrix/r5up2                 > $(T)/r5up2.out
	$(V) ./cl    < Matrix/r5up2                 > $(T)/l_r5up2.out
	$(V) ./cm    < Matrix/c5up2                 > $(T)/c5up2.out
	$(V) ./cl    < Matrix/c5up2                 > $(T)/l_c5up2.out
	$(V) ./cm    < Matrix/z5up2                 > $(T)/z5up2.out
	$(V) ./cl    < Matrix/z5up2                 > $(T)/l_z5up2.out
	$(V) ./cm -m < Matrix/z5lo                  > $(T)/z5lo.out
	$(V) ./cm    < Matrix/ibm32                 > $(T)/ibm.out
	$(V) ./cl    < Matrix/ibm32                 > $(T)/l_ibm.out
	- $(COVER)
	$(V) ./cm -m < Matrix/c5lo                  > $(T)/c5lo.out
	$(V) ./cl -m < Matrix/c5lo                  > $(T)/l_c5lo.out
	$(V) ./cm -m < Matrix/z10                   > $(T)/z10.out
	$(V) ./cl -m < Matrix/z10                   > $(T)/l_z10.out
	$(V) ./cm -m < Matrix/z5up                  > $(T)/z5up.out
	$(V) ./cl -m < Matrix/z5up                  > $(T)/l_z5up.out
	- $(COVER)
	$(V) ./cm -s < Matrix/3singular             > $(T)/3singular.out
	$(V) ./cl -s < Matrix/3singular             > $(T)/l_3singular.out
	$(V) ./cm -s < Matrix/z3singular            > $(T)/z3singular.out
	$(V) ./cl -s < Matrix/z3singular            > $(T)/l_z3singular.out
	$(V) ./cm -s < Matrix/c3singular            > $(T)/c3singular.out
	$(V) ./cl -s < Matrix/c3singular            > $(T)/l_c3singular.out
	$(V) ./cm -m < Matrix/0                     > $(T)/0.out
	$(V) ./cl -m < Matrix/0                     > $(T)/l_0.out
	$(V) ./cm -m < Matrix/afiro                 > $(T)/afiro.out
	$(V) ./cl -m < Matrix/afiro                 > $(T)/l_afiro.out
	- $(COVER)
	$(V) ./cm -m < Matrix/k01up                 > $(T)/k01up.out
	$(V) ./cl -m < Matrix/k01up                 > $(T)/l_k01up.out
	- $(COVER)
	$(V) ./cm    < Matrix/diag                  > $(T)/diag.out
	$(V) ./cl    < Matrix/diag                  > $(T)/l_diag.out
	$(V) ./cm -m < Matrix/ex5lo                 > $(T)/ex5lo.out
	$(V) ./cl -m < Matrix/ex5lo                 > $(T)/l_ex5lo.out
	- $(COVER)
	$(V) ./cm    < Matrix/20lo                  > $(T)/20lo.out
	$(V) ./cl    < Matrix/20lo                  > $(T)/l_20lo.out
	$(V) ./cm    < Matrix/z30lo                 > $(T)/z30lo.out
	$(V) ./cl    < Matrix/z30lo                 > $(T)/l_z30lo.out
	- $(COVER)
	$(V) ./cm -m < Matrix/z30up                 > $(T)/z30up.out
	$(V) ./cl -m < Matrix/z30up                 > $(T)/l_z30up.out
	$(V) ./cm    < Matrix/c10                   > $(T)/c10.out
	$(V) ./cl    < Matrix/c10                   > $(T)/l_c10.out
	$(V) ./cm    < Matrix/c30lo                 > $(T)/c30lo.out
	$(V) ./cl    < Matrix/c30lo                 > $(T)/l_c30lo.out
	- $(COVER)
	$(V) ./cm    < Matrix/C9840                 > $(T)/C9840.out
	$(V) ./cl    < Matrix/C9840                 > $(T)/l_C9840.out
	- $(COVER)
	$(V) ./cm -m < Matrix/c30up                 > $(T)/c30up.out
	$(V) ./cl -m < Matrix/c30up                 > $(T)/l_c30up.out
	$(V) ./cm    < Matrix/pi                    > $(T)/pi.out
	$(V) ./cl    < Matrix/pi                    > $(T)/l_pi.out
	$(V) ./cm    < Matrix/cpi                   > $(T)/cpi.out
	$(V) ./cl    < Matrix/cpi                   > $(T)/l_cpi.out
	$(V) ./cm    < Matrix/1_0                   > $(T)/1_0.out
	$(V) ./cl    < Matrix/1_0                   > $(T)/l_1_0.out
	$(V) ./cm -s < Matrix/3b                    > $(T)/3b.out
	$(V) ./cl -s < Matrix/3b                    > $(T)/l_3b.out
	$(V) ./cm -s < Matrix/cza                   > $(T)/cza2.out
	$(V) ./cl -s < Matrix/cza                   > $(T)/l_cza2.out
	$(V) ./cm    < Matrix/0_1                   > $(T)/0_1.out
	$(V) ./cl    < Matrix/0_1                   > $(T)/l_0_1.out
	- $(COVER)
	$(V) ./cm -n < Matrix/galenet               > $(T)/galenet_nan.out
	- $(COVER)
	$(V) ./cl -n < Matrix/galenet               > $(T)/l_galenet_nan.out
	- $(COVER)
	$(V) ./cm    < Matrix/zero                  > $(T)/zero.out
	$(V) ./cl    < Matrix/zero                  > $(T)/l_zero.out
	- $(COVER)

cov:
	- $(COVER)

# This test exercises the GPU. Do the following:
#       setenv CHOLMOD_USE_GPU 1
#       make gpu
#       setenv CHOLMOD_USE_GPU 0
#       make gpu
gpu: ldemo
	./ldemo ../../../Matrix/nd6k.mtx
	./ldemo ../../../Matrix/nd6k_complex.mtx
	./ldemo ../../../Matrix/nd6k_complex.mtx 1
	- $(COVER)

purge: clean

distclean: clean

clean:
	- $(RM) cm cl cmread clread *.c.gcov *.out zdemo ldemo
	- $(RM) leak zz_*.c z_*.c *.a l_*.c zl_*.c cov.sort ldemo.c zdemo.c
	- $(RM) -r cm.profile cmread.profile zdemo.profile $(T)
	- $(RM) -r cl.profile clread.profile ldemo.profile
	- $(RM) temp*.mtx timelog.m l_*.cu
	- $(RM) ui_*.c ul_*.c u_*.c n_*.c
	- $(RM) -r $(PURGE)
	- $(RM) -r $(CLEAN)

$(AMDOBJ): $(AMDSRC)

$(LAMDOBJ): $(AMDSRC)

$(IOBJ): $(INC)

$(LOBJ): $(INC)

.c.o:
	$(C) -c $(I) $*.c

#-------------------------------------------------------------------------------
# SUITESPARSE_CONFIG
#-------------------------------------------------------------------------------

zz_SuiteSparse_config.o: ../../SuiteSparse_config/SuiteSparse_config.c \
    ../../SuiteSparse_config/SuiteSparse_config.h
	- ln -s $< zz_SuiteSparse_config.c
	$(C) -c $(I) zz_SuiteSparse_config.c

#-------------------------------------------------------------------------------
# AMD
#-------------------------------------------------------------------------------

zz_amd_1.o: ../../AMD/Source/amd_1.c
	- ln -s $< zz_amd_1.c
	$(C) -c $(I) zz_amd_1.c

zz_amd_2.o: ../../AMD/Source/amd_2.c
	- ln -s $< zz_amd_2.c
	$(C) -c $(I) zz_amd_2.c

zz_amd_aat.o: ../../AMD/Source/amd_aat.c
	- ln -s $< zz_amd_aat.c
	$(C) -c $(I) zz_amd_aat.c

zz_amd_control.o: ../../AMD/Source/amd_control.c
	- ln -s $< zz_amd_control.c
	$(C) -c $(I) zz_amd_control.c

zz_amd_defaults.o: ../../AMD/Source/amd_defaults.c
	- ln -s $< zz_amd_defaults.c
	$(C) -c $(I) zz_amd_defaults.c

# zz_amd_dump.o: ../../AMD/Source/amd_dump.c
# 	- ln -s $< zz_amd_dump.c
# 	$(C) -c $(I) zz_amd_dump.c

zz_amd_info.o: ../../AMD/Source/amd_info.c
	- ln -s $< zz_amd_info.c
	$(C) -c $(I) zz_amd_info.c

zz_amd_order.o: ../../AMD/Source/amd_order.c
	- ln -s $< zz_amd_order.c
	$(C) -c $(I) zz_amd_order.c

zz_amd_postorder.o: ../../AMD/Source/amd_postorder.c
	- ln -s $< zz_amd_postorder.c
	$(C) -c $(I) zz_amd_postorder.c

zz_amd_post_tree.o: ../../AMD/Source/amd_post_tree.c
	- ln -s $< zz_amd_post_tree.c
	$(C) -c $(I) zz_amd_post_tree.c

zz_amd_preprocess.o: ../../AMD/Source/amd_preprocess.c
	- ln -s $< zz_amd_preprocess.c
	$(C) -c $(I) zz_amd_preprocess.c

zz_amd_valid.o: ../../AMD/Source/amd_valid.c
	- ln -s $< zz_amd_valid.c
	$(C) -c $(I) zz_amd_valid.c

#-------------------------------------------------------------------------------

zl_amd_1.o: ../../AMD/Source/amd_l1.c
	- ln -s $< zl_amd_1.c
	$(C) -c $(I) zl_amd_1.c

zl_amd_2.o: ../../AMD/Source/amd_l2.c
	- ln -s $< zl_amd_2.c
	$(C) -c $(I) zl_amd_2.c

zl_amd_aat.o: ../../AMD/Source/amd_l_aat.c
	- ln -s $< zl_amd_aat.c
	$(C) -c $(I) zl_amd_aat.c

zl_amd_control.o: ../../AMD/Source/amd_l_control.c
	- ln -s $< zl_amd_control.c
	$(C) -c $(I) zl_amd_control.c

zl_amd_defaults.o: ../../AMD/Source/amd_l_defaults.c
	- ln -s $< zl_amd_defaults.c
	$(C) -c $(I) zl_amd_defaults.c

# zl_amd_dump.o: ../../AMD/Source/amd_l_dump.c
# 	- ln -s $< zl_amd_dump.c
# 	$(C) -c $(I) zl_amd_dump.c

zl_amd_info.o: ../../AMD/Source/amd_l_info.c
	- ln -s $< zl_amd_info.c
	$(C) -c $(I) zl_amd_info.c

zl_amd_order.o: ../../AMD/Source/amd_l_order.c
	- ln -s $< zl_amd_order.c
	$(C) -c $(I) zl_amd_order.c

zl_amd_postorder.o: ../../AMD/Source/amd_l_postorder.c
	- ln -s $< zl_amd_postorder.c
	$(C) -c $(I) zl_amd_postorder.c

zl_amd_post_tree.o: ../../AMD/Source/amd_l_post_tree.c
	- ln -s $< zl_amd_post_tree.c
	$(C) -c $(I) zl_amd_post_tree.c

zl_amd_preprocess.o: ../../AMD/Source/amd_l_preprocess.c
	- ln -s $< zl_amd_preprocess.c
	$(C) -c $(I) zl_amd_preprocess.c

zl_amd_valid.o: ../../AMD/Source/amd_l_valid.c
	- ln -s $< zl_amd_valid.c
	$(C) -c $(I) zl_amd_valid.c

#-------------------------------------------------------------------------------
# CAMD
#-------------------------------------------------------------------------------

zz_camd_1.o: ../../CAMD/Source/camd_1.c
	- ln -s $< zz_camd_1.c
	$(C) -c $(I) zz_camd_1.c

zz_camd_2.o: ../../CAMD/Source/camd_2.c
	- ln -s $< zz_camd_2.c
	$(C) -c $(I) zz_camd_2.c

zz_camd_aat.o: ../../CAMD/Source/camd_aat.c
	- ln -s $< zz_camd_aat.c
	$(C) -c $(I) zz_camd_aat.c

zz_camd_control.o: ../../CAMD/Source/camd_control.c
	- ln -s $< zz_camd_control.c
	$(C) -c $(I) zz_camd_control.c

zz_camd_defaults.o: ../../CAMD/Source/camd_defaults.c
	- ln -s $< zz_camd_defaults.c
	$(C) -c $(I) zz_camd_defaults.c

# zz_camd_dump.o: ../../CAMD/Source/camd_dump.c
# 	- ln -s $< zz_camd_dump.c
# 	$(C) -c $(I) zz_camd_dump.c

zz_camd_info.o: ../../CAMD/Source/camd_info.c
	- ln -s $< zz_camd_info.c
	$(C) -c $(I) zz_camd_info.c

zz_camd_order.o: ../../CAMD/Source/camd_order.c
	- ln -s $< zz_camd_order.c
	$(C) -c $(I) zz_camd_order.c

zz_camd_postorder.o: ../../CAMD/Source/camd_postorder.c
	- ln -s $< zz_camd_postorder.c
	$(C) -c $(I) zz_camd_postorder.c

zz_camd_preprocess.o: ../../CAMD/Source/camd_preprocess.c
	- ln -s $< zz_camd_preprocess.c
	$(C) -c $(I) zz_camd_preprocess.c

zz_camd_valid.o: ../../CAMD/Source/camd_valid.c
	- ln -s $< zz_camd_valid.c
	$(C) -c $(I) zz_camd_valid.c

#-------------------------------------------------------------------------------

zl_camd_1.o: ../../CAMD/Source/camd_l1.c
	- ln -s $< zl_camd_1.c
	$(C) -c $(I) zl_camd_1.c

zl_camd_2.o: ../../CAMD/Source/camd_l2.c
	- ln -s $< zl_camd_2.c
	$(C) -c $(I) zl_camd_2.c

zl_camd_aat.o: ../../CAMD/Source/camd_l_aat.c
	- ln -s $< zl_camd_aat.c
	$(C) -c $(I) zl_camd_aat.c

zl_camd_control.o: ../../CAMD/Source/camd_l_control.c
	- ln -s $< zl_camd_control.c
	$(C) -c $(I) zl_camd_control.c

zl_camd_defaults.o: ../../CAMD/Source/camd_l_defaults.c
	- ln -s $< zl_camd_defaults.c
	$(C) -c $(I) zl_camd_defaults.c

# zl_camd_dump.o: ../../CAMD/Source/camd_l_dump.c
# 	- ln -s $< zl_camd_dump.c
# 	$(C) -c $(I) zl_camd_dump.c

zl_camd_info.o: ../../CAMD/Source/camd_l_info.c
	- ln -s $< zl_camd_info.c
	$(C) -c $(I) zl_camd_info.c

zl_camd_order.o: ../../CAMD/Source/camd_l_order.c
	- ln -s $< zl_camd_order.c
	$(C) -c $(I) zl_camd_order.c

zl_camd_postorder.o: ../../CAMD/Source/camd_l_postorder.c
	- ln -s $< zl_camd_postorder.c
	$(C) -c $(I) zl_camd_postorder.c

zl_camd_preprocess.o: ../../CAMD/Source/camd_l_preprocess.c
	- ln -s $< zl_camd_preprocess.c
	$(C) -c $(I) zl_camd_preprocess.c

zl_camd_valid.o: ../../CAMD/Source/camd_l_valid.c
	- ln -s $< zl_camd_valid.c
	$(C) -c $(I) zl_camd_valid.c

#-------------------------------------------------------------------------------

zz_colamd.o: ../../COLAMD/Source/colamd.c
	- ln -s $< zz_colamd.c
	$(C) -c $(I) zz_colamd.c

zl_colamd.o: ../../COLAMD/Source/colamd_l.c
	- ln -s $< zl_colamd.c
	$(C) -c $(I) zl_colamd.c

#-------------------------------------------------------------------------------

zz_ccolamd.o: ../../CCOLAMD/Source/ccolamd.c
	- ln -s $< zz_ccolamd.c
	$(C) -c $(I) zz_ccolamd.c

zl_ccolamd.o: ../../CCOLAMD/Source/ccolamd_l.c
	- ln -s $< zl_ccolamd.c
	$(C) -c $(I) zl_ccolamd.c

#-------------------------------------------------------------------------------

z_check.o: ../Check/cholmod_check.c
	- ln -s $< z_check.c
	$(C) -c $(I) z_check.c

z_read.o: ../Check/cholmod_read.c
	- ln -s $< z_read.c
	$(C) -c $(I) z_read.c

z_write.o: ../Check/cholmod_write.c
	- ln -s $< z_write.c
	$(C) -c $(I) z_write.c

#-------------------------------------------------------------------------------
# Utility, int32
#-------------------------------------------------------------------------------

ui_change_factor.o: ../Utility/cholmod_change_factor.c
	- ln -s $< ui_change_factor.c
	$(C) -c $(I) ui_change_factor.c

ui_transpose_unsym.o: ../Utility/cholmod_transpose_unsym.c
	- ln -s $< ui_transpose_unsym.c
	$(C) -c $(I) ui_transpose_unsym.c

ui_transpose_sym.o: ../Utility/cholmod_transpose_sym.c
	- ln -s $< ui_transpose_sym.c
	$(C) -c $(I) ui_transpose_sym.c

ui_transpose.o: ../Utility/cholmod_transpose.c
	- ln -s $< ui_transpose.c
	$(C) -c $(I) ui_transpose.c

ui_ptranspose.o: ../Utility/cholmod_ptranspose.c
	- ln -s $< ui_ptranspose.c
	$(C) -c $(I) ui_ptranspose.c

ui_sort.o: ../Utility/cholmod_sort.c
	- ln -s $< ui_sort.c
	$(C) -c $(I) ui_sort.c

ui_aat.o: ../Utility/cholmod_aat.c
	- ln -s $< ui_aat.c
	$(C) -c $(I) ui_aat.c

ui_factor_to_sparse.o: ../Utility/cholmod_factor_to_sparse.c
	- ln -s $< ui_factor_to_sparse.c
	$(C) -c $(I) ui_factor_to_sparse.c

ui_reallocate_column.o: ../Utility/cholmod_reallocate_column.c
	- ln -s $< ui_reallocate_column.c
	$(C) -c $(I) ui_reallocate_column.c

ui_copy_factor.o: ../Utility/cholmod_copy_factor.c
	- ln -s $< ui_copy_factor.c
	$(C) -c $(I) ui_copy_factor.c

ui_pack_factor.o: ../Utility/cholmod_pack_factor.c
	- ln -s $< ui_pack_factor.c
	$(C) -c $(I) ui_pack_factor.c

ui_reallocate_factor.o: ../Utility/cholmod_reallocate_factor.c
	- ln -s $< ui_reallocate_factor.c
	$(C) -c $(I) ui_reallocate_factor.c

ui_free_factor.o: ../Utility/cholmod_free_factor.c
	- ln -s $< ui_free_factor.c
	$(C) -c $(I) ui_free_factor.c

ui_alloc_factor.o: ../Utility/cholmod_alloc_factor.c
	- ln -s $< ui_alloc_factor.c
	$(C) -c $(I) ui_alloc_factor.c

ui_allocate_factor.o: ../Utility/cholmod_allocate_factor.c
	- ln -s $< ui_allocate_factor.c
	$(C) -c $(I) ui_allocate_factor.c

ui_copy.o: ../Utility/cholmod_copy.c
	- ln -s $< ui_copy.c
	$(C) -c $(I) ui_copy.c

ui_add.o: ../Utility/cholmod_add.c
	- ln -s $< ui_add.c
	$(C) -c $(I) ui_add.c

ui_add_size_t.o: ../Utility/cholmod_add_size_t.c
	- ln -s $< ui_add_size_t.c
	$(C) -c $(I) ui_add_size_t.c

ui_alloc_work.o: ../Utility/cholmod_alloc_work.c
	- ln -s $< ui_alloc_work.c
	$(C) -c $(I) ui_alloc_work.c

ui_allocate_dense.o: ../Utility/cholmod_allocate_dense.c
	- ln -s $< ui_allocate_dense.c
	$(C) -c $(I) ui_allocate_dense.c

ui_allocate_sparse.o: ../Utility/cholmod_allocate_sparse.c
	- ln -s $< ui_allocate_sparse.c
	$(C) -c $(I) ui_allocate_sparse.c

ui_allocate_triplet.o: ../Utility/cholmod_allocate_triplet.c
	- ln -s $< ui_allocate_triplet.c
	$(C) -c $(I) ui_allocate_triplet.c

ui_allocate_work.o: ../Utility/cholmod_allocate_work.c
	- ln -s $< ui_allocate_work.c
	$(C) -c $(I) ui_allocate_work.c

ui_band.o: ../Utility/cholmod_band.c
	- ln -s $< ui_band.c
	$(C) -c $(I) ui_band.c

ui_band_nnz.o: ../Utility/cholmod_band_nnz.c
	- ln -s $< ui_band_nnz.c
	$(C) -c $(I) ui_band_nnz.c

ui_calloc.o: ../Utility/cholmod_calloc.c
	- ln -s $< ui_calloc.c
	$(C) -c $(I) ui_calloc.c

ui_clear_flag.o: ../Utility/cholmod_clear_flag.c
	- ln -s $< ui_clear_flag.c
	$(C) -c $(I) ui_clear_flag.c

ui_copy_dense.o: ../Utility/cholmod_copy_dense.c
	- ln -s $< ui_copy_dense.c
	$(C) -c $(I) ui_copy_dense.c

ui_copy_dense2.o: ../Utility/cholmod_copy_dense2.c
	- ln -s $< ui_copy_dense2.c
	$(C) -c $(I) ui_copy_dense2.c

ui_copy_sparse.o: ../Utility/cholmod_copy_sparse.c
	- ln -s $< ui_copy_sparse.c
	$(C) -c $(I) ui_copy_sparse.c

ui_copy_triplet.o: ../Utility/cholmod_copy_triplet.c
	- ln -s $< ui_copy_triplet.c
	$(C) -c $(I) ui_copy_triplet.c

ui_cumsum.o: ../Utility/cholmod_cumsum.c
	- ln -s $< ui_cumsum.c
	$(C) -c $(I) ui_cumsum.c

ui_dbound.o: ../Utility/cholmod_dbound.c
	- ln -s $< ui_dbound.c
	$(C) -c $(I) ui_dbound.c

ui_defaults.o: ../Utility/cholmod_defaults.c
	- ln -s $< ui_defaults.c
	$(C) -c $(I) ui_defaults.c

ui_dense_nnz.o: ../Utility/cholmod_dense_nnz.c
	- ln -s $< ui_dense_nnz.c
	$(C) -c $(I) ui_dense_nnz.c

ui_dense_to_sparse.o: ../Utility/cholmod_dense_to_sparse.c
	- ln -s $< ui_dense_to_sparse.c
	$(C) -c $(I) ui_dense_to_sparse.c

ui_divcomplex.o: ../Utility/cholmod_divcomplex.c
	- ln -s $< ui_divcomplex.c
	$(C) -c $(I) ui_divcomplex.c

ui_ensure_dense.o: ../Utility/cholmod_ensure_dense.c
	- ln -s $< ui_ensure_dense.c
	$(C) -c $(I) ui_ensure_dense.c

ui_error.o: ../Utility/cholmod_error.c
	- ln -s $< ui_error.c
	$(C) -c $(I) ui_error.c

ui_eye.o: ../Utility/cholmod_eye.c
	- ln -s $< ui_eye.c
	$(C) -c $(I) ui_eye.c

ui_finish.o: ../Utility/cholmod_finish.c
	- ln -s $< ui_finish.c
	$(C) -c $(I) ui_finish.c

ui_free.o: ../Utility/cholmod_free.c
	- ln -s $< ui_free.c
	$(C) -c $(I) ui_free.c

ui_free_dense.o: ../Utility/cholmod_free_dense.c
	- ln -s $< ui_free_dense.c
	$(C) -c $(I) ui_free_dense.c

ui_free_sparse.o: ../Utility/cholmod_free_sparse.c
	- ln -s $< ui_free_sparse.c
	$(C) -c $(I) ui_free_sparse.c

ui_free_triplet.o: ../Utility/cholmod_free_triplet.c
	- ln -s $< ui_free_triplet.c
	$(C) -c $(I) ui_free_triplet.c

ui_free_work.o: ../Utility/cholmod_free_work.c
	- ln -s $< ui_free_work.c
	$(C) -c $(I) ui_free_work.c

ui_hypot.o: ../Utility/cholmod_hypot.c
	- ln -s $< ui_hypot.c
	$(C) -c $(I) ui_hypot.c

ui_malloc.o: ../Utility/cholmod_malloc.c
	- ln -s $< ui_malloc.c
	$(C) -c $(I) ui_malloc.c

ui_maxrank.o: ../Utility/cholmod_maxrank.c
	- ln -s $< ui_maxrank.c
	$(C) -c $(I) ui_maxrank.c

ui_mult_size_t.o: ../Utility/cholmod_mult_size_t.c
	- ln -s $< ui_mult_size_t.c
	$(C) -c $(I) ui_mult_size_t.c

ui_nnz.o: ../Utility/cholmod_nnz.c
	- ln -s $< ui_nnz.c
	$(C) -c $(I) ui_nnz.c

ui_ones.o: ../Utility/cholmod_ones.c
	- ln -s $< ui_ones.c
	$(C) -c $(I) ui_ones.c

ui_realloc.o: ../Utility/cholmod_realloc.c
	- ln -s $< ui_realloc.c
	$(C) -c $(I) ui_realloc.c

ui_realloc_multiple.o: ../Utility/cholmod_realloc_multiple.c
	- ln -s $< ui_realloc_multiple.c
	$(C) -c $(I) ui_realloc_multiple.c

ui_reallocate_sparse.o: ../Utility/cholmod_reallocate_sparse.c
	- ln -s $< ui_reallocate_sparse.c
	$(C) -c $(I) ui_reallocate_sparse.c

ui_reallocate_triplet.o: ../Utility/cholmod_reallocate_triplet.c
	- ln -s $< ui_reallocate_triplet.c
	$(C) -c $(I) ui_reallocate_triplet.c

ui_sbound.o: ../Utility/cholmod_sbound.c
	- ln -s $< ui_sbound.c
	$(C) -c $(I) ui_sbound.c

ui_score_comp.o: ../Utility/cholmod_score_comp.c
	- ln -s $< ui_score_comp.c
	$(C) -c $(I) ui_score_comp.c

ui_set_empty.o: ../Utility/cholmod_set_empty.c
	- ln -s $< ui_set_empty.c
	$(C) -c $(I) ui_set_empty.c

ui_sparse_to_dense.o: ../Utility/cholmod_sparse_to_dense.c
	- ln -s $< ui_sparse_to_dense.c
	$(C) -c $(I) ui_sparse_to_dense.c

ui_sparse_to_triplet.o: ../Utility/cholmod_sparse_to_triplet.c
	- ln -s $< ui_sparse_to_triplet.c
	$(C) -c $(I) ui_sparse_to_triplet.c

ui_speye.o: ../Utility/cholmod_speye.c
	- ln -s $< ui_speye.c
	$(C) -c $(I) ui_speye.c

ui_spzeros.o: ../Utility/cholmod_spzeros.c
	- ln -s $< ui_spzeros.c
	$(C) -c $(I) ui_spzeros.c

ui_start.o: ../Utility/cholmod_start.c
	- ln -s $< ui_start.c
	$(C) -c $(I) ui_start.c

ui_triplet_to_sparse.o: ../Utility/cholmod_triplet_to_sparse.c
	- ln -s $< ui_triplet_to_sparse.c
	$(C) -c $(I) ui_triplet_to_sparse.c

ui_version.o: ../Utility/cholmod_version.c
	- ln -s $< ui_version.c
	$(C) -c $(I) ui_version.c

ui_xtype.o: ../Utility/cholmod_xtype.c
	- ln -s $< ui_xtype.c
	$(C) -c $(I) ui_xtype.c

ui_zeros.o: ../Utility/cholmod_zeros.c
	- ln -s $< ui_zeros.c
	$(C) -c $(I) ui_zeros.c

#-------------------------------------------------------------------------------
# Utility, just one compilation
#-------------------------------------------------------------------------------

u_mult_uint64_t.o: ../Utility/cholmod_mult_uint64_t.c
	- ln -s $< u_mult_uint64_t.c
	$(C) -c $(I) u_mult_uint64_t.c

u_memdebug.o: ../Utility/cholmod_memdebug.c
	- ln -s $< u_memdebug.c
	$(C) -c $(I) u_memdebug.c

#-------------------------------------------------------------------------------
# Utility, int64
#-------------------------------------------------------------------------------

ul_change_factor.o: ../Utility/cholmod_l_change_factor.c
	- ln -s $< ul_change_factor.c
	$(C) -c $(I) ul_change_factor.c

ul_transpose_unsym.o: ../Utility/cholmod_l_transpose_unsym.c
	- ln -s $< ul_transpose_unsym.c
	$(C) -c $(I) ul_transpose_unsym.c

ul_transpose_sym.o: ../Utility/cholmod_l_transpose_sym.c
	- ln -s $< ul_transpose_sym.c
	$(C) -c $(I) ul_transpose_sym.c

ul_transpose.o: ../Utility/cholmod_l_transpose.c
	- ln -s $< ul_transpose.c
	$(C) -c $(I) ul_transpose.c

ul_ptranspose.o: ../Utility/cholmod_l_ptranspose.c
	- ln -s $< ul_ptranspose.c
	$(C) -c $(I) ul_ptranspose.c

ul_sort.o: ../Utility/cholmod_l_sort.c
	- ln -s $< ul_sort.c
	$(C) -c $(I) ul_sort.c

ul_aat.o: ../Utility/cholmod_l_aat.c
	- ln -s $< ul_aat.c
	$(C) -c $(I) ul_aat.c

ul_factor_to_sparse.o: ../Utility/cholmod_l_factor_to_sparse.c
	- ln -s $< ul_factor_to_sparse.c
	$(C) -c $(I) ul_factor_to_sparse.c

ul_reallocate_column.o: ../Utility/cholmod_l_reallocate_column.c
	- ln -s $< ul_reallocate_column.c
	$(C) -c $(I) ul_reallocate_column.c

ul_copy_factor.o: ../Utility/cholmod_l_copy_factor.c
	- ln -s $< ul_copy_factor.c
	$(C) -c $(I) ul_copy_factor.c

ul_pack_factor.o: ../Utility/cholmod_l_pack_factor.c
	- ln -s $< ul_pack_factor.c
	$(C) -c $(I) ul_pack_factor.c

ul_reallocate_factor.o: ../Utility/cholmod_l_reallocate_factor.c
	- ln -s $< ul_reallocate_factor.c
	$(C) -c $(I) ul_reallocate_factor.c

ul_free_factor.o: ../Utility/cholmod_l_free_factor.c
	- ln -s $< ul_free_factor.c
	$(C) -c $(I) ul_free_factor.c

ul_alloc_factor.o: ../Utility/cholmod_l_alloc_factor.c
	- ln -s $< ul_alloc_factor.c
	$(C) -c $(I) ul_alloc_factor.c

ul_allocate_factor.o: ../Utility/cholmod_l_allocate_factor.c
	- ln -s $< ul_allocate_factor.c
	$(C) -c $(I) ul_allocate_factor.c

ul_copy.o: ../Utility/cholmod_l_copy.c
	- ln -s $< ul_copy.c
	$(C) -c $(I) ul_copy.c

ul_add.o: ../Utility/cholmod_l_add.c
	- ln -s $< ul_add.c
	$(C) -c $(I) ul_add.c

ul_add_size_t.o: ../Utility/cholmod_l_add_size_t.c
	- ln -s $< ul_add_size_t.c
	$(C) -c $(I) ul_add_size_t.c

ul_alloc_work.o: ../Utility/cholmod_l_alloc_work.c
	- ln -s $< ul_alloc_work.c
	$(C) -c $(I) ul_alloc_work.c

ul_allocate_dense.o: ../Utility/cholmod_l_allocate_dense.c
	- ln -s $< ul_allocate_dense.c
	$(C) -c $(I) ul_allocate_dense.c

ul_allocate_sparse.o: ../Utility/cholmod_l_allocate_sparse.c
	- ln -s $< ul_allocate_sparse.c
	$(C) -c $(I) ul_allocate_sparse.c

ul_allocate_triplet.o: ../Utility/cholmod_l_allocate_triplet.c
	- ln -s $< ul_allocate_triplet.c
	$(C) -c $(I) ul_allocate_triplet.c

ul_allocate_work.o: ../Utility/cholmod_l_allocate_work.c
	- ln -s $< ul_allocate_work.c
	$(C) -c $(I) ul_allocate_work.c

ul_band.o: ../Utility/cholmod_l_band.c
	- ln -s $< ul_band.c
	$(C) -c $(I) ul_band.c

ul_band_nnz.o: ../Utility/cholmod_l_band_nnz.c
	- ln -s $< ul_band_nnz.c
	$(C) -c $(I) ul_band_nnz.c

ul_calloc.o: ../Utility/cholmod_l_calloc.c
	- ln -s $< ul_calloc.c
	$(C) -c $(I) ul_calloc.c

ul_clear_flag.o: ../Utility/cholmod_l_clear_flag.c
	- ln -s $< ul_clear_flag.c
	$(C) -c $(I) ul_clear_flag.c

ul_copy_dense.o: ../Utility/cholmod_l_copy_dense.c
	- ln -s $< ul_copy_dense.c
	$(C) -c $(I) ul_copy_dense.c

ul_copy_dense2.o: ../Utility/cholmod_l_copy_dense2.c
	- ln -s $< ul_copy_dense2.c
	$(C) -c $(I) ul_copy_dense2.c

ul_copy_sparse.o: ../Utility/cholmod_l_copy_sparse.c
	- ln -s $< ul_copy_sparse.c
	$(C) -c $(I) ul_copy_sparse.c

ul_copy_triplet.o: ../Utility/cholmod_l_copy_triplet.c
	- ln -s $< ul_copy_triplet.c
	$(C) -c $(I) ul_copy_triplet.c

ul_cumsum.o: ../Utility/cholmod_l_cumsum.c
	- ln -s $< ul_cumsum.c
	$(C) -c $(I) ul_cumsum.c

ul_dbound.o: ../Utility/cholmod_l_dbound.c
	- ln -s $< ul_dbound.c
	$(C) -c $(I) ul_dbound.c

ul_defaults.o: ../Utility/cholmod_l_defaults.c
	- ln -s $< ul_defaults.c
	$(C) -c $(I) ul_defaults.c

ul_dense_nnz.o: ../Utility/cholmod_l_dense_nnz.c
	- ln -s $< ul_dense_nnz.c
	$(C) -c $(I) ul_dense_nnz.c

ul_dense_to_sparse.o: ../Utility/cholmod_l_dense_to_sparse.c
	- ln -s $< ul_dense_to_sparse.c
	$(C) -c $(I) ul_dense_to_sparse.c

ul_divcomplex.o: ../Utility/cholmod_l_divcomplex.c
	- ln -s $< ul_divcomplex.c
	$(C) -c $(I) ul_divcomplex.c

ul_ensure_dense.o: ../Utility/cholmod_l_ensure_dense.c
	- ln -s $< ul_ensure_dense.c
	$(C) -c $(I) ul_ensure_dense.c

ul_error.o: ../Utility/cholmod_l_error.c
	- ln -s $< ul_error.c
	$(C) -c $(I) ul_error.c

ul_eye.o: ../Utility/cholmod_l_eye.c
	- ln -s $< ul_eye.c
	$(C) -c $(I) ul_eye.c

ul_finish.o: ../Utility/cholmod_l_finish.c
	- ln -s $< ul_finish.c
	$(C) -c $(I) ul_finish.c

ul_free.o: ../Utility/cholmod_l_free.c
	- ln -s $< ul_free.c
	$(C) -c $(I) ul_free.c

ul_free_dense.o: ../Utility/cholmod_l_free_dense.c
	- ln -s $< ul_free_dense.c
	$(C) -c $(I) ul_free_dense.c

ul_free_sparse.o: ../Utility/cholmod_l_free_sparse.c
	- ln -s $< ul_free_sparse.c
	$(C) -c $(I) ul_free_sparse.c

ul_free_triplet.o: ../Utility/cholmod_l_free_triplet.c
	- ln -s $< ul_free_triplet.c
	$(C) -c $(I) ul_free_triplet.c

ul_free_work.o: ../Utility/cholmod_l_free_work.c
	- ln -s $< ul_free_work.c
	$(C) -c $(I) ul_free_work.c

ul_hypot.o: ../Utility/cholmod_l_hypot.c
	- ln -s $< ul_hypot.c
	$(C) -c $(I) ul_hypot.c

ul_malloc.o: ../Utility/cholmod_l_malloc.c
	- ln -s $< ul_malloc.c
	$(C) -c $(I) ul_malloc.c

ul_maxrank.o: ../Utility/cholmod_l_maxrank.c
	- ln -s $< ul_maxrank.c
	$(C) -c $(I) ul_maxrank.c

ul_mult_size_t.o: ../Utility/cholmod_l_mult_size_t.c
	- ln -s $< ul_mult_size_t.c
	$(C) -c $(I) ul_mult_size_t.c

ul_nnz.o: ../Utility/cholmod_l_nnz.c
	- ln -s $< ul_nnz.c
	$(C) -c $(I) ul_nnz.c

ul_ones.o: ../Utility/cholmod_l_ones.c
	- ln -s $< ul_ones.c
	$(C) -c $(I) ul_ones.c

ul_realloc.o: ../Utility/cholmod_l_realloc.c
	- ln -s $< ul_realloc.c
	$(C) -c $(I) ul_realloc.c

ul_realloc_multiple.o: ../Utility/cholmod_l_realloc_multiple.c
	- ln -s $< ul_realloc_multiple.c
	$(C) -c $(I) ul_realloc_multiple.c

ul_reallocate_sparse.o: ../Utility/cholmod_l_reallocate_sparse.c
	- ln -s $< ul_reallocate_sparse.c
	$(C) -c $(I) ul_reallocate_sparse.c

ul_reallocate_triplet.o: ../Utility/cholmod_l_reallocate_triplet.c
	- ln -s $< ul_reallocate_triplet.c
	$(C) -c $(I) ul_reallocate_triplet.c

ul_sbound.o: ../Utility/cholmod_l_sbound.c
	- ln -s $< ul_sbound.c
	$(C) -c $(I) ul_sbound.c

ul_score_comp.o: ../Utility/cholmod_l_score_comp.c
	- ln -s $< ul_score_comp.c
	$(C) -c $(I) ul_score_comp.c

ul_set_empty.o: ../Utility/cholmod_l_set_empty.c
	- ln -s $< ul_set_empty.c
	$(C) -c $(I) ul_set_empty.c

ul_sparse_to_dense.o: ../Utility/cholmod_l_sparse_to_dense.c
	- ln -s $< ul_sparse_to_dense.c
	$(C) -c $(I) ul_sparse_to_dense.c

ul_sparse_to_triplet.o: ../Utility/cholmod_l_sparse_to_triplet.c
	- ln -s $< ul_sparse_to_triplet.c
	$(C) -c $(I) ul_sparse_to_triplet.c

ul_speye.o: ../Utility/cholmod_l_speye.c
	- ln -s $< ul_speye.c
	$(C) -c $(I) ul_speye.c

ul_spzeros.o: ../Utility/cholmod_l_spzeros.c
	- ln -s $< ul_spzeros.c
	$(C) -c $(I) ul_spzeros.c

ul_start.o: ../Utility/cholmod_l_start.c
	- ln -s $< ul_start.c
	$(C) -c $(I) ul_start.c

ul_triplet_to_sparse.o: ../Utility/cholmod_l_triplet_to_sparse.c
	- ln -s $< ul_triplet_to_sparse.c
	$(C) -c $(I) ul_triplet_to_sparse.c

ul_version.o: ../Utility/cholmod_l_version.c
	- ln -s $< ul_version.c
	$(C) -c $(I) ul_version.c

ul_xtype.o: ../Utility/cholmod_l_xtype.c
	- ln -s $< ul_xtype.c
	$(C) -c $(I) ul_xtype.c

ul_zeros.o: ../Utility/cholmod_l_zeros.c
	- ln -s $< ul_zeros.c
	$(C) -c $(I) ul_zeros.c

#-------------------------------------------------------------------------------

z_amd.o: ../Cholesky/cholmod_amd.c
	- ln -s $< z_amd.c
	$(C) -c $(I) z_amd.c

z_analyze.o: ../Cholesky/cholmod_analyze.c
	- ln -s $< z_analyze.c
	$(C) -c $(I) z_analyze.c

z_colamd.o: ../Cholesky/cholmod_colamd.c
	- ln -s $< z_colamd.c
	$(C) -c $(I) z_colamd.c

z_etree.o: ../Cholesky/cholmod_etree.c
	- ln -s $< z_etree.c
	$(C) -c $(I) z_etree.c

z_factorize.o: ../Cholesky/cholmod_factorize.c
	- ln -s $< z_factorize.c
	$(C) -c $(I) z_factorize.c

z_postorder.o: ../Cholesky/cholmod_postorder.c
	- ln -s $< z_postorder.c
	$(C) -c $(I) z_postorder.c

z_rcond.o: ../Cholesky/cholmod_rcond.c
	- ln -s $< z_rcond.c
	$(C) -c $(I) z_rcond.c

z_resymbol.o: ../Cholesky/cholmod_resymbol.c
	- ln -s $< z_resymbol.c
	$(C) -c $(I) z_resymbol.c

z_rowcolcounts.o: ../Cholesky/cholmod_rowcolcounts.c
	- ln -s $< z_rowcolcounts.c
	$(C) -c $(I) z_rowcolcounts.c

z_solve.o: ../Cholesky/cholmod_solve.c
	- ln -s $< z_solve.c
	$(C) -c $(I) z_solve.c

z_spsolve.o: ../Cholesky/cholmod_spsolve.c
	- ln -s $< z_spsolve.c
	$(C) -c $(I) z_spsolve.c

z_rowfac.o: ../Cholesky/cholmod_rowfac.c
	- ln -s $< z_rowfac.c
	$(C) -c $(I) z_rowfac.c

#------------------------------------------------------------------------------- 
z_ccolamd.o: ../Partition/cholmod_ccolamd.c
	- ln -s $< z_ccolamd.c
	$(C) -c $(I) z_ccolamd.c

z_csymamd.o: ../Partition/cholmod_csymamd.c
	- ln -s $< z_csymamd.c
	$(C) -c $(I) z_csymamd.c

z_camd.o: ../Partition/cholmod_camd.c
	- ln -s $< z_camd.c
	$(C) -c $(I) z_camd.c

z_metis.o: ../Partition/cholmod_metis.c
	- ln -s $< z_metis.c
	$(C) -c $(I) z_metis.c

z_nesdis.o: ../Partition/cholmod_nesdis.c
	- ln -s $< z_nesdis.c
	$(C) -c $(I) z_nesdis.c

# do not use test coverage for this file:
n_metis_wrapper.o: ../Partition/cholmod_metis_wrapper.c
	- ln -s $< n_metis_wrapper.c
	$(CN) -c $(I) n_metis_wrapper.c

#-------------------------------------------------------------------------------

z_horzcat.o: ../MatrixOps/cholmod_horzcat.c
	- ln -s $< z_horzcat.c
	$(C) -c $(I) z_horzcat.c

z_norm.o: ../MatrixOps/cholmod_norm.c
	$- ln -s $< z_norm.c
	$(C) -c $(I) z_norm.c

z_scale.o: ../MatrixOps/cholmod_scale.c
	- ln -s $< z_scale.c
	$(C) -c $(I) z_scale.c

z_drop.o: ../MatrixOps/cholmod_drop.c
	- ln -s $< z_drop.c
	$(C) -c $(I) z_drop.c

z_sdmult.o: ../MatrixOps/cholmod_sdmult.c
	- ln -s $< z_sdmult.c
	$(C) -c $(I) z_sdmult.c

z_ssmult.o: ../MatrixOps/cholmod_ssmult.c
	- ln -s $< z_ssmult.c
	$(C) -c $(I) z_ssmult.c

z_submatrix.o: ../MatrixOps/cholmod_submatrix.c
	- ln -s $< z_submatrix.c
	$(C) -c $(I) z_submatrix.c

z_vertcat.o: ../MatrixOps/cholmod_vertcat.c
	- ln -s $< z_vertcat.c
	$(C) -c $(I) z_vertcat.c

z_symmetry.o: ../MatrixOps/cholmod_symmetry.c
	- ln -s $< z_symmetry.c
	$(C) -c $(I) z_symmetry.c

#-------------------------------------------------------------------------------

z_rowadd.o: ../Modify/cholmod_rowadd.c
	- ln -s $< z_rowadd.c
	$(C) -c $(I) z_rowadd.c

z_rowdel.o: ../Modify/cholmod_rowdel.c
	- ln -s $< z_rowdel.c
	$(C) -c $(I) z_rowdel.c

z_updown.o: ../Modify/cholmod_updown.c
	- ln -s $< z_updown.c
	$(C) -c $(I) z_updown.c

#-------------------------------------------------------------------------------

z_super_numeric.o: ../Supernodal/cholmod_super_numeric.c
	- ln -s $< z_super_numeric.c
	$(C) -c $(I) z_super_numeric.c

z_super_symbolic.o: ../Supernodal/cholmod_super_symbolic.c
	- ln -s $< z_super_symbolic.c
	$(C) -c $(I) z_super_symbolic.c

z_gpu.o: ../GPU/cholmod_gpu.c
	- ln -s $< z_gpu.c
	$(C) -c $(I) z_gpu.c

z_super_solve.o: ../Supernodal/cholmod_super_solve.c
	- ln -s $< z_super_solve.c
	$(C) -c $(I) z_super_solve.c

#-------------------------------------------------------------------------------

l_check.o: ../Check/cholmod_l_check.c
	- ln -s $< l_check.c
	$(C) -c $(I) l_check.c

l_read.o: ../Check/cholmod_l_read.c
	- ln -s $< l_read.c
	$(C) -c $(I) l_read.c

l_write.o: ../Check/cholmod_l_write.c
	- ln -s $< l_write.c
	$(C) -c $(I) l_write.c

#-------------------------------------------------------------------------------

l_amd.o: ../Cholesky/cholmod_l_amd.c
	- ln -s $< l_amd.c
	$(C) -c $(I) l_amd.c

l_analyze.o: ../Cholesky/cholmod_l_analyze.c
	- ln -s $< l_analyze.c
	$(C) -c $(I) l_analyze.c

l_colamd.o: ../Cholesky/cholmod_l_colamd.c
	- ln -s $< l_colamd.c
	$(C) -c $(I) l_colamd.c

l_etree.o: ../Cholesky/cholmod_l_etree.c
	- ln -s $< l_etree.c
	$(C) -c $(I) l_etree.c

l_factorize.o: ../Cholesky/cholmod_l_factorize.c
	- ln -s $< l_factorize.c
	$(C) -c $(I) l_factorize.c

l_postorder.o: ../Cholesky/cholmod_l_postorder.c
	- ln -s $< l_postorder.c
	$(C) -c $(I) l_postorder.c

l_rcond.o: ../Cholesky/cholmod_l_rcond.c
	- ln -s $< l_rcond.c
	$(C) -c $(I) l_rcond.c

l_resymbol.o: ../Cholesky/cholmod_l_resymbol.c
	- ln -s $< l_resymbol.c
	$(C) -c $(I) l_resymbol.c

l_rowcolcounts.o: ../Cholesky/cholmod_l_rowcolcounts.c
	- ln -s $< l_rowcolcounts.c
	$(C) -c $(I) l_rowcolcounts.c

l_solve.o: ../Cholesky/cholmod_l_solve.c
	- ln -s $< l_solve.c
	$(C) -c $(I) l_solve.c

l_spsolve.o: ../Cholesky/cholmod_l_spsolve.c
	- ln -s $< l_spsolve.c
	$(C) -c $(I) l_spsolve.c

l_rowfac.o: ../Cholesky/cholmod_l_rowfac.c
	- ln -s $< l_rowfac.c
	$(C) -c $(I) l_rowfac.c

#------------------------------------------------------------------------------- 
l_ccolamd.o: ../Partition/cholmod_l_ccolamd.c
	- ln -s $< l_ccolamd.c
	$(C) -c $(I) l_ccolamd.c

l_csymamd.o: ../Partition/cholmod_l_csymamd.c
	- ln -s $< l_csymamd.c
	$(C) -c $(I) l_csymamd.c

l_camd.o: ../Partition/cholmod_l_camd.c
	- ln -s $< l_camd.c
	$(C) -c $(I) l_camd.c

l_metis.o: ../Partition/cholmod_l_metis.c
	- ln -s $< l_metis.c
	$(C) -c $(I) l_metis.c

l_nesdis.o: ../Partition/cholmod_l_nesdis.c
	- ln -s $< l_nesdis.c
	$(C) -c $(I) l_nesdis.c

#-------------------------------------------------------------------------------

l_horzcat.o: ../MatrixOps/cholmod_l_horzcat.c
	- ln -s $< l_horzcat.c
	$(C) -c $(I) l_horzcat.c

l_norm.o: ../MatrixOps/cholmod_l_norm.c
	- ln -s $< l_norm.c
	$(C) -c $(I) l_norm.c

l_scale.o: ../MatrixOps/cholmod_l_scale.c
	- ln -s $< l_scale.c
	$(C) -c $(I) l_scale.c

l_drop.o: ../MatrixOps/cholmod_l_drop.c
	- ln -s $< l_drop.c
	$(C) -c $(I) l_drop.c

l_sdmult.o: ../MatrixOps/cholmod_l_sdmult.c
	- ln -s $< l_sdmult.c
	$(C) -c $(I) l_sdmult.c

l_ssmult.o: ../MatrixOps/cholmod_l_ssmult.c
	- ln -s $< l_ssmult.c
	$(C) -c $(I) l_ssmult.c

l_submatrix.o: ../MatrixOps/cholmod_l_submatrix.c
	- ln -s $< l_submatrix.c
	$(C) -c $(I) l_submatrix.c

l_vertcat.o: ../MatrixOps/cholmod_l_vertcat.c
	- ln -s $< l_vertcat.c
	$(C) -c $(I) l_vertcat.c

l_symmetry.o: ../MatrixOps/cholmod_l_symmetry.c
	- ln -s $< l_symmetry.c
	$(C) -c $(I) l_symmetry.c

#-------------------------------------------------------------------------------

l_rowadd.o: ../Modify/cholmod_l_rowadd.c
	- ln -s $< l_rowadd.c
	$(C) -c $(I) l_rowadd.c

l_rowdel.o: ../Modify/cholmod_l_rowdel.c
	- ln -s $< l_rowdel.c
	$(C) -c $(I) l_rowdel.c

l_updown.o: ../Modify/cholmod_l_updown.c
	- ln -s $< l_updown.c
	$(C) -c $(I) l_updown.c

#-------------------------------------------------------------------------------

l_super_numeric.o: ../Supernodal/cholmod_l_super_numeric.c
	- ln -s $< l_super_numeric.c
	$(C) -c $(I) l_super_numeric.c

l_super_symbolic.o: ../Supernodal/cholmod_l_super_symbolic.c
	- ln -s $< l_super_symbolic.c
	$(C) -c $(I) l_super_symbolic.c

l_gpu.o: ../GPU/cholmod_l_gpu.c
	- ln -s $< l_gpu.c
	$(C) -c $(I) l_gpu.c

l_super_solve.o: ../Supernodal/cholmod_l_super_solve.c
	- ln -s $< l_super_solve.c
	$(C) -c $(I) l_super_solve.c

# GPU kernels only use int64_t:
l_gpu_kernels.o: ../GPU/cholmod_gpu_kernels.cu
	- ln -s $< l_gpu_kernels.cu
	$(NVCC) $(CHOLMOD_CONFIG) -I../../SuiteSparse_config -Xcompiler -fPIC -O3 -c l_gpu_kernels.cu

#-------------------------------------------------------------------------------
