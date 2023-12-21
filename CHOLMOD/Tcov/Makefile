#-------------------------------------------------------------------------------
# CHOLMOD/Tcov/Makefile
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+
#-------------------------------------------------------------------------------

# To compile and run: use "make".
# This test only works in Linux.

default: go

#-------------------------------------------------------------------------------

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
        # GPU_CONFIG must include -DCHOLMOD_HAS_CUDA to compile SuiteSparse for
        # the GPU.  You can add additional GPU-related flags to it as well.
        # with 4 cores (default):
        GPU_CONFIG    = -DCHOLMOD_HAS_CUDA
        # For example, to compile CHOLMOD for 10 CPU cores when using the GPU:
        # GPU_CONFIG  = -DCHOLMOD_HAS_CUDA
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

#-------------------------------------------------------------------------------

# without valgrind
V =
COVER = ./covall
CF = -O0 -g --coverage -fprofile-abs-path -DTEST_COVERAGE -DBLAS32 -fopenmp

# with valgrind
# V = valgrind --suppressions=suppress --quiet
# COVER =
# CF = -O0 -g -DTEST_COVERAGE -DBLAS32 -fopenmp

# Temp directory
T = $(TCOV_TMP)/T

# Tcov requires gcc
CC = gcc

# to test Tcov without METIS, but with CAMD, CCOLAMD, and CSYMAMD:
# C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(NANTESTS) -DNPARTITION

# to test with everthing
C = $(CC) $(CF) $(CHOLMOD_CONFIG) $(NANTESTS)

# no test coverage
CN = $(CC) -O0 -g -fopenmp $(CHOLMOD_CONFIG) $(NANTESTS)

# LDLIBS = -L$(SUITESPARSE)/lib -lsuitesparseconfig \
#       -lm $(LAPACK) $(BLAS) -lrt -Wl,-rpath=$(SUITESPARSE)/lib

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

ccode:  \
    di_test dl_test si_test sl_test \
    di_demo dl_demo si_demo sl_demo \
    di_read dl_read si_read sl_read

PROGS = \
    di_test dl_test si_test sl_test \
    di_demo dl_demo si_demo sl_demo \
    di_read dl_read si_read sl_read

TEMPLATES =             \
    t_amdtest.c         \
    t_aug.c             \
    t_camdtest.c        \
    t_cctest.c          \
    t_cm.c              \
    t_read_triplet.c    \
    t_rhs.c             \
    t_znorm_diag.c      \
    t_prand.c           \
    t_perm_matrix.c     \
    t_rand_dense.c      \
    t_ptest.c           \
    t_ctest.c           \
    t_huge.c            \
    t_lpdemo.c          \
    t_memory.c          \
    t_null2.c           \
    t_basic.c           \
    t_overflow_tests.c  \
    t_cat_tests.c       \
    t_null.c            \
    t_raw_factor.c      \
    t_solve.c           \
    t_test_ops.c        \
    t_test_ops2.c       \
    t_dense_tests.c     \
    t_dtype_tests.c     \
    t_common_tests.c    \
    t_error_tests.c     \
    t_tofrom_tests.c    \
    t_suitesparse.c     \
    t_unpack.c

DL_TEST = dl_test.c dl_amdtest.c dl_camdtest.c dl_huge.c
DI_TEST = di_test.c di_amdtest.c di_camdtest.c di_huge.c
SL_TEST = sl_test.c sl_amdtest.c sl_camdtest.c sl_huge.c
SI_TEST = si_test.c si_amdtest.c si_camdtest.c si_huge.c

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
        ../../AMD/Source/amd_version.c \
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
        zz_amd_valid.o \
        zz_amd_version.o \
        # add this to AMDOBJ if debugging is enabled
        # zz_amd_dump.o \

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
        zl_amd_valid.o \
        zz_amd_version.o \
        # add this to LAMDOBJ if debugging is enabled
        # zl_amd_dump.o \

COLAMDOBJ = zz_colamd.o zz_colamd_version.o

LCOLAMDOBJ = zl_colamd.o zz_colamd_version.o

#-------------------------------------------------------------------------------
# When using the Partition Module:

CCOLAMDSRC = \
        ../../CCOLAMD/Source/ccolamd.c \
        ../../CCOLAMD/Include/ccolamd.h

CCOLAMDOBJ = zz_ccolamd.o zz_ccolamd_version.o

LCCOLAMDOBJ = zl_ccolamd.o zz_ccolamd_version.o

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
        zz_camd_valid.o \
        zz_camd_version.o
        # add this to CAMDOBJ if debugging is enabled
        # zz_camd_dump.o \

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
        zl_camd_valid.o \
        zz_camd_version.o
        # add this to LCAMDOBJ if debugging is enabled
        #   zl_camd_dump.o \

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


di_test: $(IALL) $(DI_TEST) cm.h Makefile $(TEMPLATES)
	$(C) $(I) $(DI_TEST) -o di_test $(IALL) $(LDLIBS)

dl_test: $(LALL) $(DL_TEST) cm.h Makefile $(TEMPLATES)
	$(C) $(I) $(DL_TEST) -o dl_test $(LALL) $(LDLIBS)

si_test: $(IALL) $(SI_TEST) cm.h Makefile $(TEMPLATES)
	$(C) $(I) $(SI_TEST) -o si_test $(IALL) $(LDLIBS)

sl_test: $(IALL) $(SL_TEST) cm.h Makefile $(TEMPLATES)
	$(C) $(I) $(SL_TEST) -o sl_test $(LALL) $(LDLIBS)

di_read: $(IALL) di_read.c Makefile t_cmread.c t_znorm_diag.c
	$(C) $(I) di_read.c -o di_read $(IALL) $(LDLIBS)

dl_read: $(LALL) dl_read.c Makefile t_cmread.c t_znorm_diag.c
	$(C) $(I) dl_read.c -o dl_read $(LALL) $(LDLIBS)

si_read: $(IALL) si_read.c Makefile t_cmread.c t_znorm_diag.c
	$(C) $(I) si_read.c -o si_read $(IALL) $(LDLIBS)

sl_read: $(LALL) sl_read.c Makefile t_cmread.c t_znorm_diag.c
	$(C) $(I) sl_read.c -o sl_read $(LALL) $(LDLIBS)

di_demo: $(IALL) ../Demo/cholmod_di_demo.c Makefile \
	../Demo/cholmod_demo.h
	- ln -s ../Demo/cholmod_di_demo.c di_demo.c
	$(C) $(I) -I../Demo di_demo.c -o di_demo $(IALL) $(LDLIBS)

si_demo: $(IALL) ../Demo/cholmod_si_demo.c Makefile \
	../Demo/cholmod_demo.h
	- ln -s ../Demo/cholmod_si_demo.c si_demo.c
	$(C) $(I) -I../Demo si_demo.c -o si_demo $(IALL) $(LDLIBS)

dl_demo: $(LALL) ../Demo/cholmod_dl_demo.c Makefile \
	../Demo/cholmod_demo.h
	- ln -s ../Demo/cholmod_dl_demo.c dl_demo.c
	$(C) $(I) -I../Demo dl_demo.c -o dl_demo $(LALL) $(LDLIBS)

sl_demo: $(LALL) ../Demo/cholmod_sl_demo.c Makefile \
	../Demo/cholmod_demo.h
	- ln -s ../Demo/cholmod_sl_demo.c sl_demo.c
	$(C) $(I) -I../Demo sl_demo.c -o sl_demo $(LALL) $(LDLIBS)

compile:  ccode

go: ccode
	mkdir -p $(T)
	# total test time: about 22 minutes including compilation:
	$(V) ./di_read -x Matrix/crud1            > $(T)/di_read_crud1.out
	$(V) ./dl_read -x Matrix/crud1            > $(T)/dl_read_crud1.out
	$(V) ./si_read -x Matrix/crud1            > $(T)/si_read_crud1.out
	$(V) ./sl_read -x Matrix/crud1            > $(T)/sl_read_crud1.out
	#
	$(V) ./di_read -x Matrix/crud2            > $(T)/di_read_crud2.out
	$(V) ./dl_read -x Matrix/crud2            > $(T)/dl_read_crud2.out
	$(V) ./si_read -x Matrix/crud2            > $(T)/si_read_crud2.out
	$(V) ./sl_read -x Matrix/crud2            > $(T)/sl_read_crud2.out
	#
	$(V) ./di_read -x Matrix/fullcrud.mtx     > $(T)/di_read_fullcrud.out
	$(V) ./dl_read -x Matrix/fullcrud.mtx     > $(T)/dl_read_fullcrud.out
	$(V) ./si_read -x Matrix/fullcrud.mtx     > $(T)/si_read_fullcrud.out
	$(V) ./sl_read -x Matrix/fullcrud.mtx     > $(T)/sl_read_fullcrud.out
	#
	$(V) ./di_read -x Matrix/fullcrud1.mtx    > $(T)/di_read_fullcrud1.out
	$(V) ./dl_read -x Matrix/fullcrud1.mtx    > $(T)/dl_read_fullcrud1.out
	$(V) ./si_read -x Matrix/fullcrud1.mtx    > $(T)/si_read_fullcrud1.out
	$(V) ./sl_read -x Matrix/fullcrud1.mtx    > $(T)/sl_read_fullcrud1.out
	#
	$(V) ./di_read -x Matrix/fullcrud2.mtx    > $(T)/di_read_fullcrud2.out
	$(V) ./dl_read -x Matrix/fullcrud2.mtx    > $(T)/dl_read_fullcrud2.out
	$(V) ./si_read -x Matrix/fullcrud2.mtx    > $(T)/si_read_fullcrud2.out
	$(V) ./sl_read -x Matrix/fullcrud2.mtx    > $(T)/sl_read_fullcrud2.out
	#
	$(V) ./di_read Matrix/3by0.mtx            > $(T)/di_read_3by0.out
	$(V) ./dl_read Matrix/3by0.mtx            > $(T)/dl_read_3by0.out
	$(V) ./si_read Matrix/3by0.mtx            > $(T)/si_read_3by0.out
	$(V) ./sl_read Matrix/3by0.mtx            > $(T)/sl_read_3by0.out
	#
	$(V) ./di_read Matrix/fullrza.mtx         > $(T)/di_read_fullrza.out
	$(V) ./dl_read Matrix/fullrza.mtx         > $(T)/dl_read_fullrza.out
	$(V) ./si_read Matrix/fullrza.mtx         > $(T)/si_read_fullrza.out
	$(V) ./sl_read Matrix/fullrza.mtx         > $(T)/sl_read_fullrza.out
	#
	$(V) ./di_read Matrix/fullrsa.mtx         > $(T)/di_read_fullrsa.out
	$(V) ./dl_read Matrix/fullrsa.mtx         > $(T)/dl_read_fullrsa.out
	$(V) ./si_read Matrix/fullrsa.mtx         > $(T)/si_read_fullrsa.out
	$(V) ./sl_read Matrix/fullrsa.mtx         > $(T)/sl_read_fullrsa.out
	#
	$(V) ./di_read Matrix/fullcsa.mtx         > $(T)/di_read_fullcsa.out
	$(V) ./dl_read Matrix/fullcsa.mtx         > $(T)/dl_read_fullcsa.out
	$(V) ./si_read Matrix/fullcsa.mtx         > $(T)/si_read_fullcsa.out
	$(V) ./sl_read Matrix/fullcsa.mtx         > $(T)/sl_read_fullcsa.out
	#
	$(V) ./di_read Matrix/cha.mtx             > $(T)/di_read_cha.out
	$(V) ./dl_read Matrix/cha.mtx             > $(T)/dl_read_cha.out
	$(V) ./si_read Matrix/cha.mtx             > $(T)/si_read_cha.out
	$(V) ./sl_read Matrix/cha.mtx             > $(T)/sl_read_cha.out
	#
	$(V) ./di_read Matrix/rza.mtx             > $(T)/di_read_rza.out
	$(V) ./dl_read Matrix/rza.mtx             > $(T)/dl_read_rza.out
	$(V) ./si_read Matrix/rza.mtx             > $(T)/si_read_rza.out
	$(V) ./sl_read Matrix/rza.mtx             > $(T)/sl_read_rza.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle1.mtx > $(T)/di_read_mangle1.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle1.mtx > $(T)/dl_read_mangle1.out
	$(V) ./si_read -x ../Demo/Matrix/mangle1.mtx > $(T)/si_read_mangle1.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle1.mtx > $(T)/sl_read_mangle1.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle2.mtx > $(T)/di_read_mangle2.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle2.mtx > $(T)/dl_read_mangle2.out
	$(V) ./si_read -x ../Demo/Matrix/mangle2.mtx > $(T)/si_read_mangle2.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle2.mtx > $(T)/sl_read_mangle2.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle3.mtx > $(T)/di_read_mangle3.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle3.mtx > $(T)/dl_read_mangle3.out
	$(V) ./si_read -x ../Demo/Matrix/mangle3.mtx > $(T)/si_read_mangle3.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle3.mtx > $(T)/sl_read_mangle3.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle4.mtx > $(T)/di_read_mangle4.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle4.mtx > $(T)/dl_read_mangle4.out
	$(V) ./si_read -x ../Demo/Matrix/mangle4.mtx > $(T)/si_read_mangle4.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle4.mtx > $(T)/sl_read_mangle4.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle5.tri > $(T)/di_read_mangle5.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle5.tri > $(T)/dl_read_mangle5.out
	$(V) ./si_read -x ../Demo/Matrix/mangle5.tri > $(T)/si_read_mangle5.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle5.tri > $(T)/sl_read_mangle5.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle6.tri > $(T)/di_read_mangle6.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle6.tri > $(T)/dl_read_mangle6.out
	$(V) ./si_read -x ../Demo/Matrix/mangle6.tri > $(T)/si_read_mangle6.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle6.tri > $(T)/sl_read_mangle6.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle7.tri > $(T)/di_read_mangle7.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle7.tri > $(T)/dl_read_mangle7.out
	$(V) ./si_read -x ../Demo/Matrix/mangle7.tri > $(T)/si_read_mangle7.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle7.tri > $(T)/sl_read_mangle7.out
	#
	$(V) ./di_read -x ../Demo/Matrix/mangle8.tri > $(T)/di_read_mangle8.out
	$(V) ./dl_read -x ../Demo/Matrix/mangle8.tri > $(T)/dl_read_mangle8.out
	$(V) ./si_read -x ../Demo/Matrix/mangle8.tri > $(T)/si_read_mangle8.out
	$(V) ./sl_read -x ../Demo/Matrix/mangle8.tri > $(T)/sl_read_mangle8.out
	#
	$(V) ./di_read Matrix/2diag.tri           > $(T)/di_read_2diag.out
	$(V) ./dl_read Matrix/2diag.tri           > $(T)/dl_read_2diag.out
	$(V) ./si_read Matrix/2diag.tri           > $(T)/si_read_2diag.out
	$(V) ./sl_read Matrix/2diag.tri           > $(T)/sl_read_2diag.out
	#
	$(V) ./di_read Matrix/r5lo2               > $(T)/di_read_r5lo2.out
	$(V) ./dl_read Matrix/r5lo2               > $(T)/dl_read_r5lo2.out
	$(V) ./si_read Matrix/r5lo2               > $(T)/si_read_r5lo2.out
	$(V) ./sl_read Matrix/r5lo2               > $(T)/sl_read_r5lo2.out
	#
	$(V) ./di_read Matrix/2lo.tri             > $(T)/di_read_2lo.out
	$(V) ./dl_read Matrix/2lo.tri             > $(T)/dl_read_2lo.out
	$(V) ./si_read Matrix/2lo.tri             > $(T)/si_read_2lo.out
	$(V) ./sl_read Matrix/2lo.tri             > $(T)/sl_read_2lo.out
	#
	$(V) ./di_read Matrix/2.tri               > $(T)/di_read_2.out
	$(V) ./dl_read Matrix/2.tri               > $(T)/dl_read_2.out
	$(V) ./si_read Matrix/2.tri               > $(T)/si_read_2.out
	$(V) ./sl_read Matrix/2.tri               > $(T)/sl_read_2.out
	#
	$(V) ./di_read Matrix/2up.tri             > $(T)/di_read_2up.out
	$(V) ./dl_read Matrix/2up.tri             > $(T)/dl_read_2up.out
	$(V) ./si_read Matrix/2up.tri             > $(T)/si_read_2up.out
	$(V) ./sl_read Matrix/2up.tri             > $(T)/sl_read_2up.out
	#
	$(V) ./di_read Matrix/1e99                > $(T)/di_read_1e99.out
	$(V) ./dl_read Matrix/1e99                > $(T)/dl_read_1e99.out
	$(V) ./si_read Matrix/1e99                > $(T)/si_read_1e99.out
	$(V) ./sl_read Matrix/1e99                > $(T)/sl_read_1e99.out
	$(COVER)
	# time: - coverage: ~4910 demo bcsstk01:
	$(V) ./di_demo   ../Demo/Matrix/bcsstk01.tri  > $(T)/di_demo_k1.out
	$(V) ./dl_demo   ../Demo/Matrix/bcsstk01.tri  > $(T)/dl_demo_k1.out
	$(V) ./si_demo   ../Demo/Matrix/bcsstk01.tri  > $(T)/si_demo_k1.out
	$(V) ./sl_demo   ../Demo/Matrix/bcsstk01.tri  > $(T)/sl_demo_k1.out
	$(COVER)
	# time: - coverage: ~2550 demo bcsstk02:
	$(V) ./di_demo   ../Demo/Matrix/bcsstk02.tri  > $(T)/di_demo_k2.out
	$(V) ./dl_demo   ../Demo/Matrix/bcsstk02.tri  > $(T)/dl_demo_k2.out
	$(V) ./si_demo   ../Demo/Matrix/bcsstk02.tri  > $(T)/si_demo_k2.out
	$(V) ./sl_demo   ../Demo/Matrix/bcsstk02.tri  > $(T)/sl_demo_k2.out
	$(COVER)
	# time: - coverage: ~1560 demo d:
	$(V) ./di_demo < ../Demo/Matrix/d.tri         > $(T)/di_demo_d.out
	$(V) ./dl_demo < ../Demo/Matrix/d.tri         > $(T)/dl_demo_d.out
	$(V) ./dl_demo   ../Demo/Matrix/d.tri 1       > $(T)/dl_demo_d.out
	$(V) ./si_demo < ../Demo/Matrix/d.tri         > $(T)/si_demo_d.out
	$(V) ./sl_demo < ../Demo/Matrix/d.tri         > $(T)/sl_demo_d.out
	$(V) ./sl_demo   ../Demo/Matrix/d.tri 1       > $(T)/sl_demo_d.out
	$(COVER)
	# time: - coverage: 30 demo up:
	$(V) ./di_demo < ../Demo/Matrix/up.tri        > $(T)/di_demo_up.out
	$(V) ./dl_demo < ../Demo/Matrix/up.tri        > $(T)/dl_demo_up.out
	$(V) ./dl_demo   ../Demo/Matrix/up.tri 1      > $(T)/dl_demo_up_z.out
	$(V) ./si_demo < ../Demo/Matrix/up.tri        > $(T)/si_demo_up.out
	$(V) ./sl_demo < ../Demo/Matrix/up.tri        > $(T)/sl_demo_up.out
	$(V) ./sl_demo   ../Demo/Matrix/up.tri 1      > $(T)/sl_demo_up_z.out
	$(COVER)
	# time: - coverage: 4 demo 3_2:
	$(V) ./di_demo < Matrix/3_2                   > $(T)/di_demo_3_2.out
	$(V) ./dl_demo < Matrix/3_2                   > $(T)/dl_demo_3_2.out
	$(V) ./si_demo < Matrix/3_2                   > $(T)/si_demo_3_2.out
	$(V) ./sl_demo < Matrix/3_2                   > $(T)/sl_demo_3_2.out
	$(COVER)
	# time: 8 coverage: ~19300  matrix: c5up2
	$(V) ./di_test -e14 < Matrix/c5up2    > $(T)/di_test_c5up2.out
	$(V) ./dl_test -e14 < Matrix/c5up2    > $(T)/dl_test_c5up2.out
	$(V) ./si_test -e5  < Matrix/c5up2    > $(T)/si_test_c5up2.out
	$(V) ./sl_test -e5  < Matrix/c5up2    > $(T)/sl_test_c5up2.out
	$(COVER)
	# time: 0 coverage: ~350   matrix: zero:
	$(V) ./di_test    < Matrix/zero       > $(T)/di_test_zero.out
	$(V) ./dl_test    < Matrix/zero       > $(T)/dl_test_zero.out
	$(V) ./si_test    < Matrix/zero       > $(T)/si_test_zero.out
	$(V) ./sl_test    < Matrix/zero       > $(T)/sl_test_zero.out
	$(COVER)
	# time: 6 coverage: ~850 cza:
	$(V) ./di_test -s < Matrix/cza        > $(T)/di_test_cza.out
	$(V) ./dl_test -s < Matrix/cza        > $(T)/dl_test_cza.out
	$(V) ./si_test -s < Matrix/cza        > $(T)/si_test_cza.out
	$(V) ./sl_test -s < Matrix/cza        > $(T)/sl_test_cza.out
	$(COVER)
	# time: 16 coverage: ~3170 z5up:
	$(V) ./di_test -m -e14 < Matrix/z5up  > $(T)/di_test_z5up.out
	$(V) ./dl_test -m -e14 < Matrix/z5up  > $(T)/dl_test_z5up.out
	$(V) ./si_test -m -e5  < Matrix/z5up  > $(T)/si_test_z5up.out
	$(V) ./sl_test -m -e5  < Matrix/z5up  > $(T)/sl_test_z5up.out
	$(COVER)
	# time: 11 coverage: ~8180 write:
	$(V) ./di_test -e5 < Matrix/write.tri > $(T)/di_test_write.out
	$(V) ./dl_test -e5 < Matrix/write.tri > $(T)/dl_test_write.out
	$(COVER)
	# time: 13 coverage: ~2640  pi:
	$(V) ./di_test -e11 < Matrix/pi       > $(T)/di_test_pi.out
	$(V) ./dl_test -e11 < Matrix/pi       > $(T)/dl_test_pi.out
	$(V) ./si_test -e2  < Matrix/pi       > $(T)/si_test_pi.out
	$(V) ./sl_test -e2  < Matrix/pi       > $(T)/sl_test_pi.out
	$(COVER)
	# time: 20 coverage: ~5150 r5lo:
	$(V) ./di_test -e11 < Matrix/r5lo     > $(T)/di_test_r5lo.out
	$(V) ./dl_test -e11 < Matrix/r5lo     > $(T)/dl_test_r5lo.out
	$(V) ./si_test -e2  < Matrix/r5lo     > $(T)/si_test_r5lo.out
	$(V) ./sl_test -e2  < Matrix/r5lo     > $(T)/sl_test_r5lo.out
	$(COVER)
	# time: 25 coverage: ~1760 5by50:
	$(V) ./di_test -e9  < Matrix/5by50    > $(T)/di_test_5by50.out
	$(V) ./dl_test -e9  < Matrix/5by50    > $(T)/dl_test_5by50.out
	$(V) ./si_test -e3  < Matrix/5by50    > $(T)/si_test_5by50.out
	$(V) ./sl_test -e3  < Matrix/5by50    > $(T)/sl_test_5by50.out
	$(COVER)
	# time: 25 coverage: ~1090 diag:
	$(V) ./di_test -e11 < Matrix/diag     > $(T)/di_test_diag.out
	$(V) ./dl_test -e11 < Matrix/diag     > $(T)/dl_test_diag.out
	$(V) ./si_test -e2  < Matrix/diag     > $(T)/si_test_diag.out
	$(V) ./sl_test -e2  < Matrix/diag     > $(T)/sl_test_diag.out
	$(COVER)
	# time: 17 coverage: 52 0:
	$(V) ./di_test -m -e14 < Matrix/0     > $(T)/di_test_0.out
	$(V) ./dl_test -m -e14 < Matrix/0     > $(T)/dl_test_0.out
	$(V) ./si_test -m -e5  < Matrix/0     > $(T)/si_test_0.out
	$(V) ./sl_test -m -e5  < Matrix/0     > $(T)/sl_test_0.out
	$(COVER)
	# time: 66 coverage: ~1900 ex50lo:
	$(V) ./di_test -m -e7 < Matrix/ex5lo  > $(T)/di_test_ex5lo.out
	$(V) ./dl_test -m -e7 < Matrix/ex5lo  > $(T)/dl_test_ex5lo.out
	$(V) ./si_test -m -e3 < Matrix/ex5lo  > $(T)/si_test_ex5lo.out
	$(V) ./sl_test -m -e3 < Matrix/ex5lo  > $(T)/sl_test_ex5lo.out
	$(COVER)
	# time: 20 coverage: 22 c10:
	$(V) ./di_test -e12 < Matrix/c10      > $(T)/di_test_c10.out
	$(V) ./dl_test -e12 < Matrix/c10      > $(T)/dl_test_c10.out
	$(V) ./si_test -e4  < Matrix/c10      > $(T)/si_test_c10.out
	$(V) ./sl_test -e4  < Matrix/c10      > $(T)/sl_test_c10.out
	$(COVER)
	# time: 46 coverage: 224 z30lo:
	$(V) ./di_test -e12   < Matrix/z30lo  > $(T)/di_test_z30lo.out
	$(V) ./dl_test -e12   < Matrix/z30lo  > $(T)/dl_test_z30lo.out
	$(V) ./si_test -e4    < Matrix/z30lo  > $(T)/si_test_z30lo.out
	$(V) ./sl_test -e4    < Matrix/z30lo  > $(T)/sl_test_z30lo.out
	$(COVER)
	# time: 40 coverage: 168 c30lo_singular:
	$(V) ./di_test -s < Matrix/c30lo_singular > $(T)/di_test_c30lo.out
	$(V) ./dl_test -s < Matrix/c30lo_singular > $(T)/dl_test_c30lo.out
	$(V) ./si_test -s < Matrix/c30lo_singular > $(T)/si_test_c30lo.out
	$(V) ./sl_test -s < Matrix/c30lo_singular > $(T)/sl_test_c30lo.out
	$(COVER)
	# time: 46 coverage: 14 c30up:
	$(V) ./di_test -e12 < Matrix/c30up > $(T)/di_test_c30up.out
	$(V) ./dl_test -e12 < Matrix/c30up > $(T)/dl_test_c30up.out
	$(V) ./si_test -e4  < Matrix/c30up > $(T)/si_test_c30up.out
	$(V) ./sl_test -e4  < Matrix/c30up > $(T)/sl_test_c30up.out
	$(COVER)
	# time: 58 coverage: 18 z10:
	$(V) ./di_test -m -e11 < Matrix/z10   > $(T)/di_test_z10.out
	$(V) ./dl_test -m -e11 < Matrix/z10   > $(T)/dl_test_z10.out
	$(V) ./si_test -m -e4  < Matrix/z10   > $(T)/si_test_z10.out
	$(V) ./sl_test -m -e4  < Matrix/z10   > $(T)/sl_test_z10.out
	$(COVER)
	# time: 60 coverage: 18 galenet:
	$(V) ./dl_test -e8 < Matrix/galenet   > $(T)/dl_test_galenet.out
	$(V) ./di_test -e8 < Matrix/galenet   > $(T)/di_test_galenet.out
	$(V) ./si_test -e2 < Matrix/galenet   > $(T)/si_test_galenet.out
	$(V) ./sl_test -e2 < Matrix/galenet   > $(T)/sl_test_galenet.out
	$(COVER)
	# time: 76 coverage: 44 20lo:
	$(V) ./di_test -e12 < Matrix/20lo     > $(T)/di_test_20lo.out
	$(V) ./dl_test -e12 < Matrix/20lo     > $(T)/dl_test_20lo.out
	$(V) ./si_test -e3  < Matrix/20lo     > $(T)/si_test_20lo.out
	$(V) ./sl_test -e3  < Matrix/20lo     > $(T)/sl_test_20lo.out
	$(COVER)
	# time: 112 coverage: 400 ibm32:
	$(V) ./di_test -e8  < Matrix/ibm32    > $(T)/di_test_ibm.out
	$(V) ./dl_test -e8  < Matrix/ibm32    > $(T)/dl_test_ibm.out
	$(V) ./si_test      < Matrix/ibm32    > $(T)/si_test_ibm.out
	$(V) ./sl_test      < Matrix/ibm32    > $(T)/sl_test_ibm.out
	$(COVER)
	# time: 106 coverage: 10 C9840:
	$(V) ./di_test -e11 < Matrix/C9840    > $(T)/di_test_C9840.out
	$(V) ./dl_test -e11 < Matrix/C9840    > $(T)/dl_test_C9840.out
	$(V) ./si_test -e2  < Matrix/C9840    > $(T)/si_test_C9840.out
	$(V) ./sl_test -e2  < Matrix/C9840    > $(T)/sl_test_C9840.out
	$(COVER)
	# time: 205 coverage: 32 afiro:
	$(V) ./di_test -m -e7  < Matrix/afiro > $(T)/di_test_afiro.out
	$(V) ./dl_test -m -e7  < Matrix/afiro > $(T)/dl_test_afiro.out
	$(V) ./si_test -m -e2  < Matrix/afiro > $(T)/si_test_afiro.out
	$(V) ./sl_test -m -e2  < Matrix/afiro > $(T)/sl_test_afiro.out
	$(COVER)
	# time: 260 coverage: 6 k01up:
	$(V) ./di_test -m -e10 < Matrix/k01up > $(T)/di_test_k01up.out
	$(V) ./dl_test -m -e10 < Matrix/k01up > $(T)/dl_test_k01up.out
	$(V) ./si_test -m -e4  < Matrix/k01up > $(T)/si_test_k01up.out
	$(V) ./sl_test -m -e4  < Matrix/k01up > $(T)/sl_test_k01up.out
	$(COVER)

cov:
	- $(COVER)

# This test exercises the GPU. Do the following:
#       setenv CHOLMOD_USE_GPU 1
#       make gpu
#       setenv CHOLMOD_USE_GPU 0
#       make gpu
gpu: dl_demo
	./dl_demo ../../../Matrix/nd6k.mtx
	./dl_demo ../../../Matrix/nd6k_complex.mtx
	./dl_demo ../../../Matrix/nd6k_complex.mtx 1
	- $(COVER)

purge: clean

distclean: clean

clean:
	- $(RM) $(PROGS)
	- $(RM) *.c.gcov *.out
	- $(RM) leak zz_*.c z_*.c *.a l_*.c zl_*.c cov.sort
	- $(RM) -r *.profile $(T)
	- $(RM) temp*.mtx timelog.m l_*.cu
	- $(RM) si_demo.c sl_demo.c
	- $(RM) di_demo.c dl_demo.c
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

zz_amd_version.o: ../../AMD/Source/amd_version.c
	- ln -s $< zz_amd_version.c
	$(C) -c $(I) zz_amd_version.c

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

zz_camd_version.o: ../../CAMD/Source/camd_version.c
	- ln -s $< zz_camd_version.c
	$(C) -c $(I) zz_camd_version.c

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

zz_colamd_version.o: ../../COLAMD/Source/colamd_version.c
	- ln -s $< zz_colamd_version.c
	$(C) -c $(I) zz_colamd_version.c

#-------------------------------------------------------------------------------

zz_ccolamd.o: ../../CCOLAMD/Source/ccolamd.c
	- ln -s $< zz_ccolamd.c
	$(C) -c $(I) zz_ccolamd.c

zl_ccolamd.o: ../../CCOLAMD/Source/ccolamd_l.c
	- ln -s $< zl_ccolamd.c
	$(C) -c $(I) zl_ccolamd.c

zz_ccolamd_version.o: ../../CCOLAMD/Source/ccolamd_version.c
	- ln -s $< zz_ccolamd_version.c
	$(C) -c $(I) zz_ccolamd_version.c

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
	- ln -s $< z_norm.c
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

#-------------------------------------------------------------------------------

# GPU kernels only use int64_t:
l_gpu_kernels.o: ../GPU/cholmod_gpu_kernels.cu
	- ln -s $< l_gpu_kernels.cu
	$(NVCC) $(CHOLMOD_CONFIG) -I../../SuiteSparse_config -Xcompiler -fPIC -O3 -c l_gpu_kernels.cu

