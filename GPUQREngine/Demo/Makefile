#-------------------------------------------------------------------------------
# GPUQREngine/Demo Makefile
#-------------------------------------------------------------------------------

include ../../SuiteSparse_config/SuiteSparse_config.mk

LIBS = $(CUDART_LIB) ../Lib/libGPUQREngine.a \
    ../../SuiteSparse_GPURuntime/Lib/libSuiteSparse_GPURuntime.a \
    ../../CHOLMOD/Lib/libcholmod.a \
    ../../SuiteSparse_config/libsuitesparseconfig.a \
    $(CUDART_LIB) $(CUBLAS_LIB) -lm -lrt

INC  = -I../../SuiteSparse_GPURuntime/Include -I../Include \
       -I../../SuiteSparse_config -I../../CHOLMOD/Include

#-------------------------------------------------------------------------------

all: lib demo

library: lib

clean:
	$(RM) *.o

purge: clean
	$(RM) gpuqrengine_demo troll.m
 
lib:
	( cd ../Lib ; $(MAKE) library )
	( cd ../../CHOLMOD ; $(MAKE) library )
	( cd ../../SuiteSparse_config ; $(MAKE) library )
	( cd ../../SuiteSparse_GPURuntime/Lib ; $(MAKE) library )

demo: gpuqrengine_demo
	./gpuqrengine_demo

gpuqrengine_demo: lib
	$(NVCC) $(INC) -o gpuqrengine_demo gpuqrengine_demo.cpp $(LIBS)


