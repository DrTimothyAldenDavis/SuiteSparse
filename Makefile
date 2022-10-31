#-------------------------------------------------------------------------------
# Makefile for all SuiteSparse packages
#-------------------------------------------------------------------------------

# edit this variable to pass options to cmake:
export CMAKE_OPTIONS ?=

# edit this variable to control parallel make:
export JOBS ?= 8

# use "CUDA = no" to disable CUDA:
export CUDA ?= auto
ifneq ($(CUDA),no)
    CUDA_PATH = $(shell which nvcc 2>/dev/null | sed "s/\/bin\/nvcc//")
else
    CUDA_PATH =
endif
export CUDA_PATH

# do not modify this variable
export SUITESPARSE = $(CURDIR)

#-------------------------------------------------------------------------------

# Compile the default rules for each package.
# "make install" will install all libraries in /usr/local/lib
# and include files in /usr/local/include.
default:
	( cd SuiteSparse_config && $(MAKE) )
	- ( cd SuiteSparse_metis && $(MAKE) )
	( cd Mongoose && $(MAKE) )
	( cd AMD && $(MAKE) )
	( cd BTF && $(MAKE) )
	( cd CAMD && $(MAKE) )
	( cd CCOLAMD && $(MAKE) )
	( cd COLAMD && $(MAKE) )
	( cd CHOLMOD && $(MAKE) )
	( cd CSparse && $(MAKE) )
	( cd CXSparse && $(MAKE) )
	( cd LDL && $(MAKE) )
	( cd KLU && $(MAKE) )
	( cd UMFPACK && $(MAKE) )
	( cd RBio && $(MAKE) )
	( cd SuiteSparse_GPURuntime && $(MAKE) )
	( cd GPUQREngine && $(MAKE) )
	( cd SPQR && $(MAKE) )
	( cd GraphBLAS && $(MAKE) )
	( cd SLIP_LU && $(MAKE) )

# compile and install in SuiteSparse/lib and SuiteSparse/include
local:
	( cd SuiteSparse_config && $(MAKE) local && $(MAKE) install )
	- ( cd SuiteSparse_metis && $(MAKE) local && $(MAKE) install )
	( cd Mongoose && $(MAKE) local && $(MAKE) install )
	( cd AMD && $(MAKE) local && $(MAKE) install )
	( cd BTF && $(MAKE) local && $(MAKE) install )
	( cd CAMD && $(MAKE) local && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) local && $(MAKE) install )
	( cd COLAMD && $(MAKE) local && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) local && $(MAKE) install )
	( cd CSparse && $(MAKE) )  # CSparse is compiled but not installed
	( cd CXSparse && $(MAKE) local && $(MAKE) install )
	( cd LDL && $(MAKE) local && $(MAKE) install )
	( cd KLU && $(MAKE) local && $(MAKE) install )
	( cd UMFPACK && $(MAKE) local && $(MAKE) install )
	( cd RBio && $(MAKE) local && $(MAKE) install )
	( cd SuiteSparse_GPURuntime && $(MAKE) local && $(MAKE) install )
	( cd GPUQREngine && $(MAKE) local && $(MAKE) install )
	( cd SPQR && $(MAKE) local && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) local && $(MAKE) install )
	( cd SLIP_LU && $(MAKE) )

# install all packages in SuiteSparse/lib and SuiteSparse/include
install: gbinstall moninstall
	( cd SuiteSparse_config && $(MAKE) install )
	- ( cd SuiteSparse_metis && $(MAKE) install )
	# ( cd Mongoose  && $(MAKE) install )
	( cd AMD && $(MAKE) install )
	( cd BTF && $(MAKE) install )
	( cd CAMD && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) install )
	( cd COLAMD && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) install )
	( cd CSparse && $(MAKE) ) # CSparse is compiled but not installed
	( cd CXSparse && $(MAKE) install ) # CXSparse is installed instead
	( cd LDL && $(MAKE) install )
	( cd KLU && $(MAKE) install )
	( cd UMFPACK && $(MAKE) install )
	( cd RBio && $(MAKE) install )
	( cd SuiteSparse_GPURuntime && $(MAKE) install )
	( cd GPUQREngine && $(MAKE) install )
	( cd SPQR && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) install )
	( cd SLIP_LU && $(MAKE) install )

# uninstall all packages
uninstall:
	( cd SuiteSparse_config && $(MAKE) uninstall )
	- ( cd SuiteSparse_metis && $(MAKE) uninstall )
	- ( cd Mongoose  && $(MAKE) uninstall )
	( cd AMD && $(MAKE) uninstall )
	( cd CAMD && $(MAKE) uninstall )
	( cd COLAMD && $(MAKE) uninstall )
	( cd BTF && $(MAKE) uninstall )
	( cd KLU && $(MAKE) uninstall )
	( cd LDL && $(MAKE) uninstall )
	( cd CCOLAMD && $(MAKE) uninstall )
	( cd UMFPACK && $(MAKE) uninstall )
	( cd CHOLMOD && $(MAKE) uninstall )
	( cd CSparse && $(MAKE) uninstall )
	( cd CXSparse && $(MAKE) uninstall )
	( cd RBio && $(MAKE) uninstall )
	( cd SuiteSparse_GPURuntime && $(MAKE) uninstall )
	( cd GPUQREngine && $(MAKE) uninstall )
	( cd SPQR && $(MAKE) uninstall )
	- ( cd GraphBLAS && $(MAKE) uninstall )
	( cd SLIP_LU && $(MAKE) uninstall )

# compile the libraries
library:
	( cd SuiteSparse_config && $(MAKE) )
	- ( cd SuiteSparse_metis && $(MAKE) library )
	( cd Mongoose  && $(MAKE) library )
	( cd AMD && $(MAKE) library )
	( cd BTF && $(MAKE) library )
	( cd CAMD && $(MAKE) library )
	( cd CCOLAMD && $(MAKE) library )
	( cd COLAMD && $(MAKE) library )
	( cd CHOLMOD && $(MAKE) library )
	( cd KLU && $(MAKE) library )
	( cd LDL && $(MAKE) library )
	( cd UMFPACK && $(MAKE) library )
	( cd CSparse && $(MAKE) library )
	( cd CXSparse && $(MAKE) library )
	( cd RBio && $(MAKE) library )
	( cd SuiteSparse_GPURuntime && $(MAKE) library )
	( cd GPUQREngine && $(MAKE) library )
	( cd SPQR && $(MAKE) library )
	( cd GraphBLAS && $(MAKE) library )
	( cd SLIP_LU && $(MAKE) library )

# Remove all files not in the original distribution
purge:
	- ( cd SuiteSparse_config && $(MAKE) purge )
	- ( cd SuiteSparse_metis && $(MAKE) purge )
	- ( cd AMD && $(MAKE) purge )
	- ( cd Mongoose  && $(MAKE) purge )
	- ( cd CAMD && $(MAKE) purge )
	- ( cd COLAMD && $(MAKE) purge )
	- ( cd BTF && $(MAKE) purge )
	- ( cd KLU && $(MAKE) purge )
	- ( cd LDL && $(MAKE) purge )
	- ( cd CCOLAMD && $(MAKE) purge )
	- ( cd UMFPACK && $(MAKE) purge )
	- ( cd CHOLMOD && $(MAKE) purge )
	- ( cd CSparse && $(MAKE) purge )
	- ( cd CXSparse && $(MAKE) purge )
	- ( cd RBio && $(MAKE) purge )
	- ( cd SuiteSparse_GPURuntime && $(MAKE) purge )
	- ( cd GPUQREngine && $(MAKE) purge )
	- ( cd SPQR && $(MAKE) purge )
	- ( cd GraphBLAS && $(MAKE) purge )
	- $(RM) MATLAB_Tools/*/*.mex* MATLAB_Tools/*/*/*.mex*
	- $(RM) MATLAB_Tools/*/*.o    MATLAB_Tools/*/*/*.o
	- $(RM) -r include/* bin/* lib/* share/*
	- ( cd SLIP_LU && $(MAKE) purge )

# Remove all files not in the original distribution, but keep the libraries
clean:
	- ( cd SuiteSparse_config && $(MAKE) clean )
	- ( cd SuiteSparse_metis && $(MAKE) clean )
	- ( cd Mongoose  && $(MAKE) clean )
	- ( cd AMD && $(MAKE) clean )
	- ( cd CAMD && $(MAKE) clean )
	- ( cd COLAMD && $(MAKE) clean )
	- ( cd BTF && $(MAKE) clean )
	- ( cd KLU && $(MAKE) clean )
	- ( cd LDL && $(MAKE) clean )
	- ( cd CCOLAMD && $(MAKE) clean )
	- ( cd UMFPACK && $(MAKE) clean )
	- ( cd CHOLMOD && $(MAKE) clean )
	- ( cd CSparse && $(MAKE) clean )
	- ( cd CXSparse && $(MAKE) clean )
	- ( cd RBio && $(MAKE) clean )
	- ( cd SuiteSparse_GPURuntime && $(MAKE) clean )
	- ( cd GPUQREngine && $(MAKE) clean )
	- ( cd SPQR && $(MAKE) clean )
	- ( cd GraphBLAS && $(MAKE) clean )
	- ( cd SLIP_LU && $(MAKE) clean )

# Create the PDF documentation
docs:
	( cd GraphBLAS && $(MAKE) docs )
#	( cd Mongoose  && $(MAKE) docs )
	( cd AMD && $(MAKE) docs )
	( cd CAMD && $(MAKE) docs )
	( cd KLU && $(MAKE) docs )
	( cd LDL && $(MAKE) docs )
	( cd UMFPACK && $(MAKE) docs )
	( cd CHOLMOD && $(MAKE) docs )
	( cd SPQR && $(MAKE) docs )
	( cd SLIP_LU && $(MAKE) docs )

distclean: purge

# statement coverage (Linux only); this requires a lot of time.
cov: purge
	( cd CXSparse && $(MAKE) cov )
	( cd CSparse && $(MAKE) cov )
	( cd CHOLMOD && $(MAKE) cov )
	( cd KLU && $(MAKE) cov )
	( cd SPQR && $(MAKE) cov )
	( cd UMFPACK && $(MAKE) cov )
	( cd SLIP_LU && $(MAKE) cov )

# just compile GraphBLAS
gb:
	( cd GraphBLAS && $(MAKE) )

# compile and install GraphBLAS
gbinstall: gb
	( cd GraphBLAS && $(MAKE) install )

# compile and install GraphBLAS libgraphblas_renamed, for MATLAB
gbmatlab:
	( cd GraphBLAS/GraphBLAS && $(MAKE) )
	( cd GraphBLAS/GraphBLAS && $(MAKE) install )

# just compile Mongoose
mon:
	( cd Mongoose && $(MAKE) )

# compile and install Mongoose
moninstall: mon
	( cd Mongoose  && $(MAKE) install )

