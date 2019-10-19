#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime/Lib/Makfile
#-------------------------------------------------------------------------------

LIBRARY = libSuiteSparse_GPURuntime
VERSION = 1.0.1

default: all

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

all: library

runtime: library

lib: library

library: $(AR_TARGET) $(SO_TARGET)

clean:
	$(RM) -r $(CLEAN)

purge: clean
	$(RM) -r $(PURGE)

distclean: purge

$(AR_TARGET): $(OBJS)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $@

$(SO_TARGET): $(OBJS)
	$(CC) $(SO_OPTS) $^ -o $@

#-------------------------------------------------------------------------------

SuiteSparseGPU_Workspace.o: ../Source/SuiteSparseGPU_Workspace.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

SuiteSparseGPU_Workspace_cpuAllocators.o: ../Source/SuiteSparseGPU_Workspace_cpuAllocators.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

SuiteSparseGPU_Workspace_gpuAllocators.o: ../Source/SuiteSparseGPU_Workspace_gpuAllocators.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

SuiteSparseGPU_Workspace_transfer.o: ../Source/SuiteSparseGPU_Workspace_transfer.cpp $(H)
	$(NVCC) -c $(INC) $(NVCCFLAGS) $<

#-------------------------------------------------------------------------------

# install SuiteSparse_GPURuntime (just the library, not the include files)
install: uninstall $(SO_TARGET)
	$(CP) $(SO_TARGET) $(INSTALL_LIB)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	chmod 644 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_LIB)/$(SO_PLAIN)

# uninstall SuiteSparse_GPURuntime
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)

