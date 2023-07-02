#===============================================================================
# SuiteSparseQR/Tcov/Makefile
#===============================================================================

# SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

# This test requires METIS, and it only works on Linux.

# run statement-coverage test without Valgrind
default: go

# to run with Valgrind as well
valgrind: vgo

ccode: all

CUDA = no

################################################################################

    SUITESPARSE ?= $(realpath $(CURDIR)/../..)
    GPU_CONFIG =

    # remove object files, but keep compiled libraries via 'make clean'
    CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d \
        *.gcda *.gcno *.aux *.bbl *.blg *.log *.toc *.dvi *.lof *.lot

    # also remove compiled libraries, via 'make distclean'
    PURGE = *.so* *.a *.dll *.dylib *.dSYM

################################################################################

BLAS = -lblas
LAPACK = -llapack

# this test requires gcc
CC = gcc
CXX= g++

# for statement coverage (with gcov; see go) and picky compiler warnings
CF = -pg -g --coverage -fprofile-abs-path -O0 \
	-Wall -W -Wshadow -Winline -Wno-unused-parameter \
	-Wredundant-decls -Wdisabled-optimization \
	-fexceptions -fopenmp -DTEST_COVERAGE -DBLAS32

# Using an optimized BLAS can cause problems in Valgrind
# FLIB = -llapack_plain -lblas_plain -lgfortran
# FLIB = -llapack_plain -lblas_plain -lgfortran -lg2c
  FLIB = $(LAPACK) $(BLAS)

#-------------------------------------------------------------------------------

CUDA_LIB = $(CUDART_LIB) $(CUBLAS_LIB)

NVCC = /usr/local/cuda/bin/nvcc -g --profile --generate-line-info $(NV20) \
	-Xcompiler -fprofile-arcs -Xcompiler -ftest-coverage

#-------------------------------------------------------------------------------

CLIB = -lsuitesparseconfig -lcholmod -lamd -lcolamd -lccolamd -lcamd

all: qrtest gpudemo qrdemo_gpu

library: qrtest gpudemo

purge: distclean

distclean: clean
	- $(RM) qrtest qrtest_out.txt pfile tfile cov.out qrtest_out1.txt
	- $(RM) gpuqrengine_demo troll.m qrdemo_gpu gpu_results.txt X.mtx
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp

OBJ = \
    SuiteSparseQR_C.o                        \
    SuiteSparseQR_expert.o                   \
    SuiteSparseQR.o                          \
    SuiteSparseQR_qmult.o                    \
    spqr_type.o                              \
    spqr_cholmod_wrappers.o                  \
    qrtestc.o                                \

ifneq ($(GPU_CONFIG),)
OBJ += \
    spqrgpu_kernel.o                         \
    spqrgpu_buildAssemblyMaps.o              \
    spqrgpu_computeFrontStaging.o            \
    SuiteSparseGPU_Workspace.o               \
    SuiteSparseGPU_Workspace_cpuAllocators.o \
    SuiteSparseGPU_Workspace_gpuAllocators.o \
    SuiteSparseGPU_Workspace_transfer.o      \
    GPUQREngine_GraphVizHelper.o             \
    GPUQREngine_UberKernel.o                 \
    GPUQREngine_ExpertSparse.o               \
    GPUQREngine_Internal.o                   \
    GPUQREngine_ExpertDense.o                \
    ssgpu_maxQueueSize.o                     \
    TaskDescriptor_flops.o                   \
    ssgpu_maxQueueSize.o
endif

$(OBJ): $(INC)

I = -I../../CHOLMOD/Include -I../../SuiteSparse_config -I../Include \

ifneq ($(GPU_CONFIG),)
    I += -I../../SuiteSparse_GPURuntime/Include \
        -I../../GPUQREngine/Include $(CUDA_INC)
endif

C = $(CXX) $(CF) $(I) $(GPU_CONFIG)

LIBS = -L../../lib $(FLIB) $(CLIB) $(CUDA_LIB) \
    -Wl,--rpath=$(SUITESPARSE)/lib -lm -lrt

qrtestc.o: qrtestc.c $(INC)
	$(CC) $(CF) -std=c11 $(I) -c $<

qrtest: qrtest.cpp $(INC) $(OBJ)
	$(C) qrtest.cpp -o qrtest $(OBJ) $(LIBS) -lm

ifneq ($(GPU_CONFIG),)
gpu: gpuqrengine_demo qrdemo_gpu
	- ./gpuqrengine_demo
	- ./qrdemo_gpu ../Matrix/west0067.mtx 2
	- ./qrdemo_gpu ../Matrix/lp_e226_transposed.mtx 2
	- ./qrdemo_gpu ../Matrix/lp_e226_transposed.mtx 6
	- ./qrdemo_gpu ../Matrix/Groebner_id2003_aug.mtx 6
	- ./qrdemo_gpu ../Matrix/Franz6_id1959_aug.mtx 6
else
gpu:
endif

gpuqrengine_demo: \
        ../../GPUQREngine/Demo/gpuqrengine_demo.cpp $(INC) $(OBJ)
	$(C) $(GPUQRDEMO)/gpuqrengine_demo.cpp -o gpuqrengine_demo \
                $(OBJ) $(LIBS) -lm

qrdemo_gpu: ../Demo/qrdemo_gpu.cpp $(INC) $(OBJ)
ifneq ($(GPU_CONFIG),)
	$(C) ../Demo/qrdemo_gpu.cpp -o qrdemo_gpu $(OBJ) $(LIBS)
endif

go: qrtest gpu qrdemo_gpu
	- ./qrtest matrixlist.txt > qrtest_out.txt
	- ./cov

go1: qrtest
	- ./qrtest matrix1.txt > qrtest_out1.txt
	- ./cov

vgo1: qrtest
	- valgrind ./qrtest matrix1.txt > qrtest_out1.txt
	# - valgrind --leak-check=full --show-reachable=yes ./qrtest matrix1.txt > qrtest_out1.txt
	- ./cov

vgo: qrtest
	- valgrind --leak-check=full --show-reachable=yes ./qrtest matrixlist.txt > qrtest_out.txt
	- ./cov

spqr_type.o: ../Source/spqr_type.cpp
	$(C) -c $<

spqr_cholmod_wrappers.o: ../Source/spqr_cholmod_wrappers.cpp
	$(C) -c $<

SuiteSparseQR_expert.o: ../Source/SuiteSparseQR_expert.cpp
	$(C) -c $<

SuiteSparseQR_qmult.o: ../Source/SuiteSparseQR_qmult.cpp
	$(C) -c $<

SuiteSparseQR.o: ../Source/SuiteSparseQR.cpp
	$(C) -c $<

SuiteSparseQR_C.o: ../Source/SuiteSparseQR_C.cpp
	$(C) -c $<

spqrgpu_kernel.o: ../SPQRGPU/spqrgpu_kernel.cpp
	$(C) -c $<

spqrgpu_buildAssemblyMaps.o: ../SPQRGPU/spqrgpu_buildAssemblyMaps.cpp
	$(C) -c $<

spqrgpu_computeFrontStaging.o: ../SPQRGPU/spqrgpu_computeFrontStaging.cpp
	$(C) -c $<

#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime
#-------------------------------------------------------------------------------

GPURUNTIME = ../../SuiteSparse_GPURuntime
GPURUNSRC = $(GPURUNTIME)/Source
GPURUNINC = -I$(GPURUNTIME)/Include -I../../SuiteSparse_config

RUNH = \
        $(GPURUNTIME)/Include/SuiteSparseGPU_debug.hpp \
        $(GPURUNTIME)/Include/SuiteSparseGPU_workspace_macros.hpp \
        $(GPURUNTIME)/Include/SuiteSparseGPU_internal.hpp \
        $(GPURUNTIME)/Include/SuiteSparse_GPURuntime.hpp

SuiteSparseGPU_Workspace.o: $(GPURUNSRC)/SuiteSparseGPU_Workspace.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_cpuAllocators.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_cpuAllocators.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_gpuAllocators.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_gpuAllocators.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_memset.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_memset.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

SuiteSparseGPU_Workspace_transfer.o: \
        $(GPURUNSRC)/SuiteSparseGPU_Workspace_transfer.cpp $(RUNH)
	$(NVCC) -c $(GPURUNINC) $<

#-------------------------------------------------------------------------------
# GPUQREngine
#-------------------------------------------------------------------------------

GPUQR = ../../GPUQREngine
GPUQRSRC = $(GPUQR)/Source
GPUQRDEMO = $(GPUQR)/Demo
GPUQRINC = $(GPURUNINC) -I$(GPUQR)/Include

KERNELH = \
    $(GPUQR)/Include/GPUQREngine_Common.hpp \
    $(GPUQR)/Include/GPUQREngine_BucketList.hpp \
    $(GPUQR)/Include/GPUQREngine_Front.hpp \
    $(GPUQR)/Include/GPUQREngine_FrontState.hpp \
    $(GPUQR)/Include/GPUQREngine_SuiteSparse.hpp \
    $(GPUQR)/Include/GPUQREngine.hpp \
    $(GPUQR)/Include/GPUQREngine_Internal.hpp \
    $(GPUQR)/Include/GPUQREngine_GraphVizHelper.hpp \
    $(GPUQR)/Include/Kernel/Apply/block_apply_1_by_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_2_by_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_2.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_3_by_1.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_3.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply_chunk.cu \
    $(GPUQR)/Include/Kernel/Apply/block_apply.cu \
    $(GPUQR)/Include/Kernel/Apply/cevta_tile.cu \
    $(GPUQR)/Include/Kernel/Apply/pipelined_rearrange.cu \
    $(GPUQR)/Include/Kernel/Assemble/packAssemble.cu \
    $(GPUQR)/Include/Kernel/Assemble/sAssemble.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_3_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_1_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_1_by_1_edge.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_2_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_2_by_1_edge.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_3_by_1.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt_3_by_1_edge.cu \
    $(GPUQR)/Include/Kernel/Factorize/factorize_vt.cu \
    $(GPUQR)/Include/Kernel/qrKernel.cu \
    $(GPUQR)/Include/Kernel/sharedMemory.hpp \
    $(GPUQR)/Include/Kernel/uberKernel.cu \
    $(GPUQR)/Include/GPUQREngine_LLBundle.hpp \
    $(GPUQR)/Include/GPUQREngine_Stats.hpp \
    $(GPUQR)/Include/GPUQREngine_Scheduler.hpp \
    $(GPUQR)/Include/GPUQREngine_SEntry.hpp \
    $(GPUQR)/Include/GPUQREngine_SparseMeta.hpp \
    $(GPUQR)/Include/GPUQREngine_TaskDescriptor.hpp \
    $(GPUQR)/Include/GPUQREngine_Timing.hpp

GPUQREngine_GraphVizHelper.o: \
        $(GPUQRSRC)/GPUQREngine_GraphVizHelper.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_UberKernel.o: $(GPUQRSRC)/GPUQREngine_UberKernel.cu $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_ExpertDense.o: $(GPUQRSRC)/GPUQREngine_ExpertDense.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_Internal.o: $(GPUQRSRC)/GPUQREngine_Internal.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

GPUQREngine_ExpertSparse.o: $(GPUQRSRC)/GPUQREngine_ExpertSparse.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

ssgpu_maxQueueSize.o: $(GPUQRSRC)/Scheduler/ssgpu_maxQueueSize.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<

TaskDescriptor_flops.o: $(GPUQRSRC)/TaskDescriptor/TaskDescriptor_flops.cpp $(KERNELH)
	$(NVCC) -c  $(GPUQRINC) $<
