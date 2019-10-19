#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime/Lib/Makfile
#-------------------------------------------------------------------------------

include ../../SuiteSparse_config/SuiteSparse_config.mk

OBJS = SuiteSparseGPU_Workspace.o                     \
       SuiteSparseGPU_Workspace_cpuAllocators.o       \
       SuiteSparseGPU_Workspace_gpuAllocators.o       \
       SuiteSparseGPU_Workspace_transfer.o

INC = -I../Include -I../../SuiteSparse_config
LIBS = $(CUDART_LIB)

H = \
        ../Include/SuiteSparseGPU_Workspace.hpp \
        ../Include/SuiteSparseGPU_debug.hpp \
        ../Include/SuiteSparseGPU_macros.hpp \
        ../Include/SuiteSparseGPU_workspace_macros.hpp \
        ../Include/SuiteSparseGPU_Runtime.hpp \
        Makefile

#-------------------------------------------------------------------------------

all: runtime

library: runtime

lib: runtime

clean:
	$(RM) -f *.o

purge: clean
	$(RM) -f libSuiteSparse_GPURuntime.a

runtime: $(OBJS)
	$(AR) -r "libSuiteSparse_GPURuntime.a" $(OBJS)

SuiteSparseGPU_Workspace.o: ../Source/SuiteSparseGPU_Workspace.cpp $(H)
	$(NVCC) -c $(INC) $<

SuiteSparseGPU_Workspace_cpuAllocators.o: ../Source/SuiteSparseGPU_Workspace_cpuAllocators.cpp $(H)
	$(NVCC) -c $(INC) $<

SuiteSparseGPU_Workspace_gpuAllocators.o: ../Source/SuiteSparseGPU_Workspace_gpuAllocators.cpp $(H)
	$(NVCC) -c $(INC) $<

SuiteSparseGPU_Workspace_transfer.o: ../Source/SuiteSparseGPU_Workspace_transfer.cpp $(H)
	$(NVCC) -c $(INC) $<

