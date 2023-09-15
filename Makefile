#-------------------------------------------------------------------------------
# Makefile for all SuiteSparse packages
#-------------------------------------------------------------------------------

# edit this variable to pass options to cmake:
export CMAKE_OPTIONS ?=

# edit this variable to control parallel make:
export JOBS ?= 8

# do not modify this variable
export SUITESPARSE = $(CURDIR)

#-------------------------------------------------------------------------------

# Compile the default rules for each package.

# default: "make install" will install all libraries in /usr/local/lib
# and include files in /usr/local/include.  Not installed in SuiteSparse/lib.
default: library

# compile; "sudo make install" will install only in /usr/local
# (or whatever your CMAKE_INSTALL_PREFIX is)
library:
	( cd SuiteSparse_config && $(MAKE) )
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
	( cd SPEX && $(MAKE) )

# compile; "make install" only in  SuiteSparse/lib and SuiteSparse/include
local:
	( cd SuiteSparse_config && $(MAKE) local )
	( cd Mongoose && $(MAKE) local )
	( cd AMD && $(MAKE) local )
	( cd BTF && $(MAKE) local )
	( cd CAMD && $(MAKE) local )
	( cd CCOLAMD && $(MAKE) local )
	( cd COLAMD && $(MAKE) local )
	( cd CHOLMOD && $(MAKE) local )
	( cd CSparse && $(MAKE) )  # CSparse is compiled but not installed
	( cd CXSparse && $(MAKE) local )
	( cd LDL && $(MAKE) local )
	( cd KLU && $(MAKE) local )
	( cd UMFPACK && $(MAKE) local )
	( cd RBio && $(MAKE) local )
	( cd SuiteSparse_GPURuntime && $(MAKE) local )
	( cd GPUQREngine && $(MAKE) local )
	( cd SPQR && $(MAKE) local )
	( cd GraphBLAS && $(MAKE) local )
	( cd SPEX && $(MAKE) local )

# compile; "sudo make install" will install only in /usr/local
# (or whatever your CMAKE_INSTALL_PREFIX is)
global:
	( cd SuiteSparse_config && $(MAKE) global )
	( cd Mongoose && $(MAKE) global )
	( cd AMD && $(MAKE) global )
	( cd BTF && $(MAKE) global )
	( cd CAMD && $(MAKE) global )
	( cd CCOLAMD && $(MAKE) global )
	( cd COLAMD && $(MAKE) global )
	( cd CHOLMOD && $(MAKE) global )
	( cd CSparse && $(MAKE) )  # CSparse is compiled but not installed
	( cd CXSparse && $(MAKE) global )
	( cd LDL && $(MAKE) global )
	( cd KLU && $(MAKE) global )
	( cd UMFPACK && $(MAKE) global )
	( cd RBio && $(MAKE) global )
	( cd SuiteSparse_GPURuntime && $(MAKE) global )
	( cd GPUQREngine && $(MAKE) global )
	( cd SPQR && $(MAKE) global )
	( cd GraphBLAS && $(MAKE) global )
	( cd SPEX && $(MAKE) global )

# install all packages.  Location depends on prior "make", "make global" etc
install:
	( cd SuiteSparse_config && $(MAKE) install )
	( cd Mongoose  && $(MAKE) install )
	( cd AMD && $(MAKE) install )
	( cd BTF && $(MAKE) install )
	( cd CAMD && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) install )
	( cd COLAMD && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) install )
	( cd CXSparse && $(MAKE) install ) # CXSparse is installed instead
	( cd LDL && $(MAKE) install )
	( cd KLU && $(MAKE) install )
	( cd UMFPACK && $(MAKE) install )
	( cd RBio && $(MAKE) install )
	( cd SuiteSparse_GPURuntime && $(MAKE) install )
	( cd GPUQREngine && $(MAKE) install )
	( cd SPQR && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) install )
	( cd SPEX && $(MAKE) install )

# uninstall all packages
uninstall:
	( cd SuiteSparse_config && $(MAKE) uninstall )
	( cd Mongoose  && $(MAKE) uninstall )
	( cd AMD && $(MAKE) uninstall )
	( cd CAMD && $(MAKE) uninstall )
	( cd COLAMD && $(MAKE) uninstall )
	( cd BTF && $(MAKE) uninstall )
	( cd KLU && $(MAKE) uninstall )
	( cd LDL && $(MAKE) uninstall )
	( cd CCOLAMD && $(MAKE) uninstall )
	( cd UMFPACK && $(MAKE) uninstall )
	( cd CHOLMOD && $(MAKE) uninstall )
	( cd CXSparse && $(MAKE) uninstall )
	( cd RBio && $(MAKE) uninstall )
	( cd SuiteSparse_GPURuntime && $(MAKE) uninstall )
	( cd GPUQREngine && $(MAKE) uninstall )
	( cd SPQR && $(MAKE) uninstall )
	( cd GraphBLAS && $(MAKE) uninstall )
	( cd SPEX && $(MAKE) uninstall )

# Remove all files not in the original distribution
distclean: purge

# Remove all files not in the original distribution
purge:
	- ( cd SuiteSparse_config && $(MAKE) purge )
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
	- $(RM) MATLAB_Tools/*/*.mex* MATLAB_Tools/*/*/*.mex*
	- $(RM) MATLAB_Tools/*/*.o    MATLAB_Tools/*/*/*.o
	- $(RM) -r Example/build/*
	- ( cd GraphBLAS && $(MAKE) purge )
	- ( cd SPEX && $(MAKE) purge )
	- $(RM) -r include/* bin/* lib/*

clean: purge

# Run all demos
demos:
	- ( cd SuiteSparse_config && $(MAKE) demos )
	- ( cd Mongoose && $(MAKE) demos )
	- ( cd AMD && $(MAKE) demos )
	- ( cd CAMD && $(MAKE) demos )
	- ( cd COLAMD && $(MAKE) demos )
	- ( cd BTF && $(MAKE) demos )
	- ( cd KLU && $(MAKE) demos )
	- ( cd LDL && $(MAKE) demos )
	- ( cd CCOLAMD && $(MAKE) demos )
	- ( cd UMFPACK && $(MAKE) demos )
	- ( cd CHOLMOD && $(MAKE) demos )
	- ( cd CSparse && $(MAKE) demos )
	- ( cd CXSparse && $(MAKE) demos )
	- ( cd RBio && $(MAKE) demos )
	- ( cd SPQR && $(MAKE) demos )
	- ( cd GraphBLAS && $(MAKE) demos )
	- ( cd SPEX && $(MAKE) demos )

# Create the PDF documentation
docs:
	( cd GraphBLAS && $(MAKE) docs )
	( cd Mongoose  && $(MAKE) docs )
	( cd AMD && $(MAKE) docs )
	( cd CAMD && $(MAKE) docs )
	( cd KLU && $(MAKE) docs )
	( cd LDL && $(MAKE) docs )
	( cd UMFPACK && $(MAKE) docs )
	( cd CHOLMOD && $(MAKE) docs )
	( cd SPQR && $(MAKE) docs )
	( cd SPEX && $(MAKE) docs )

# statement coverage (Linux only); this requires a lot of time.
cov: local install
	( cd CXSparse && $(MAKE) cov )
	( cd CSparse && $(MAKE) cov )
	( cd CHOLMOD && $(MAKE) cov )
	( cd KLU && $(MAKE) cov )
	( cd SPQR && $(MAKE) cov )
	( cd UMFPACK && $(MAKE) cov )
	( cd SPEX && $(MAKE) cov )

gbmatlab:
	( cd GraphBLAS/GraphBLAS && $(MAKE) )

gblocal:
	( cd GraphBLAS/GraphBLAS && $(MAKE) local && $(MAKE) install )

debug:
	( cd SuiteSparse_config && $(MAKE) debug )
	# ( cd Mongoose && $(MAKE) debug )
	( cd AMD && $(MAKE) debug )
	( cd BTF && $(MAKE) debug )
	( cd CAMD && $(MAKE) debug )
	( cd CCOLAMD && $(MAKE) debug )
	( cd COLAMD && $(MAKE) debug )
	( cd CHOLMOD && $(MAKE) debug )
	( cd CSparse && $(MAKE) debug )
	( cd CXSparse && $(MAKE) debug )
	( cd LDL && $(MAKE) debug )
	( cd KLU && $(MAKE) debug )
	( cd UMFPACK && $(MAKE) debug )
	( cd RBio && $(MAKE) debug )
	( cd SuiteSparse_GPURuntime && $(MAKE) )
	( cd GPUQREngine && $(MAKE) )
	( cd SPQR && $(MAKE) debug )
	( cd GraphBLAS && $(MAKE) cdebug )
	( cd SPEX && $(MAKE) debug )

