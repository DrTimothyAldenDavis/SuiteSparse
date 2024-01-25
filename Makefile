#-------------------------------------------------------------------------------
# Makefile for all SuiteSparse packages
#-------------------------------------------------------------------------------

# Copyright (c) 2023, Timothy A. Davis, All Rights Reserved.
# Just this particular file is under the Apache-2.0 license; each package has
# its own license.
# SPDX-License-Identifier: Apache-2.0

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
	( cd AMD && $(MAKE) )
	( cd COLAMD && $(MAKE) )
	( cd SPEX && $(MAKE) )
	( cd Mongoose && $(MAKE) )
	( cd BTF && $(MAKE) )
	( cd CAMD && $(MAKE) )
	( cd CCOLAMD && $(MAKE) )
	( cd CHOLMOD && $(MAKE) )
	( cd CSparse && $(MAKE) )
	( cd CXSparse && $(MAKE) )
	( cd LDL && $(MAKE) )
	( cd KLU && $(MAKE) )
	( cd UMFPACK && $(MAKE) )
	( cd ParU && $(MAKE) )
	( cd RBio && $(MAKE) )
	( cd SPQR && $(MAKE) )
	( cd GraphBLAS && $(MAKE) )
	( cd LAGraph && $(MAKE) )

# compile; "make install" only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd SuiteSparse_config && $(MAKE) local )
	( cd AMD && $(MAKE) local )
	( cd COLAMD && $(MAKE) local )
	( cd SPEX && $(MAKE) local )
	( cd Mongoose && $(MAKE) local )
	( cd BTF && $(MAKE) local )
	( cd CAMD && $(MAKE) local )
	( cd CCOLAMD && $(MAKE) local )
	( cd CHOLMOD && $(MAKE) local )
	( cd CSparse && $(MAKE) )  
# CSparse is compiled but not installed
	( cd CXSparse && $(MAKE) local )
	( cd LDL && $(MAKE) local )
	( cd KLU && $(MAKE) local )
	( cd UMFPACK && $(MAKE) local )
	( cd ParU && $(MAKE) local )
	( cd RBio && $(MAKE) local )
	( cd SPQR && $(MAKE) local )
	( cd GraphBLAS && $(MAKE) local )
	( cd LAGraph && $(MAKE) local )

# compile; "sudo make install" will install only in /usr/local
# (or whatever your CMAKE_INSTALL_PREFIX is)
global:
	( cd SuiteSparse_config && $(MAKE) global )
	( cd AMD && $(MAKE) global )
	( cd COLAMD && $(MAKE) global )
	( cd SPEX && $(MAKE) global )
	( cd Mongoose && $(MAKE) global )
	( cd BTF && $(MAKE) global )
	( cd CAMD && $(MAKE) global )
	( cd CCOLAMD && $(MAKE) global )
	( cd CHOLMOD && $(MAKE) global )
	( cd CSparse && $(MAKE) )  
# CSparse is compiled but not installed
	( cd CXSparse && $(MAKE) global )
	( cd LDL && $(MAKE) global )
	( cd KLU && $(MAKE) global )
	( cd UMFPACK && $(MAKE) global )
	( cd ParU && $(MAKE) global )
	( cd RBio && $(MAKE) global )
	( cd SPQR && $(MAKE) global )
	( cd GraphBLAS && $(MAKE) global )
	( cd LAGraph && $(MAKE) global )

# install all packages.  Location depends on prior "make", "make global" etc
install:
	( cd SuiteSparse_config && $(MAKE) install )
	( cd AMD && $(MAKE) install )
	( cd COLAMD && $(MAKE) install )
	( cd SPEX && $(MAKE) install )
	( cd Mongoose  && $(MAKE) install )
	( cd BTF && $(MAKE) install )
	( cd CAMD && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) install )
	( cd CXSparse && $(MAKE) install ) 
# CXSparse is installed instead
	( cd LDL && $(MAKE) install )
	( cd KLU && $(MAKE) install )
	( cd UMFPACK && $(MAKE) install )
	( cd ParU && $(MAKE) install )
	( cd RBio && $(MAKE) install )
	( cd SPQR && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) install )
	( cd LAGraph && $(MAKE) install )

# uninstall all packages
uninstall:
	( cd SuiteSparse_config && $(MAKE) uninstall )
	( cd AMD && $(MAKE) uninstall )
	( cd COLAMD && $(MAKE) uninstall )
	( cd SPEX && $(MAKE) uninstall )
	( cd Mongoose  && $(MAKE) uninstall )
	( cd CAMD && $(MAKE) uninstall )
	( cd BTF && $(MAKE) uninstall )
	( cd KLU && $(MAKE) uninstall )
	( cd LDL && $(MAKE) uninstall )
	( cd CCOLAMD && $(MAKE) uninstall )
	( cd ParU && $(MAKE) uninstall )
	( cd UMFPACK && $(MAKE) uninstall )
	( cd CHOLMOD && $(MAKE) uninstall )
	( cd CXSparse && $(MAKE) uninstall )
	( cd RBio && $(MAKE) uninstall )
	( cd SPQR && $(MAKE) uninstall )
	( cd GraphBLAS && $(MAKE) uninstall )
	( cd LAGraph && $(MAKE) uninstall )

# Remove all files not in the original distribution
distclean: purge

# Remove all files not in the original distribution
purge:
	- ( cd SuiteSparse_config && $(MAKE) purge )
	- ( cd AMD && $(MAKE) purge )
	- ( cd COLAMD && $(MAKE) purge )
	- ( cd SPEX && $(MAKE) purge )
	- ( cd Mongoose  && $(MAKE) purge )
	- ( cd CAMD && $(MAKE) purge )
	- ( cd BTF && $(MAKE) purge )
	- ( cd KLU && $(MAKE) purge )
	- ( cd LDL && $(MAKE) purge )
	- ( cd CCOLAMD && $(MAKE) purge )
	- ( cd UMFPACK && $(MAKE) purge )
	- ( cd CHOLMOD && $(MAKE) purge )
	- ( cd CSparse && $(MAKE) purge )
	- ( cd CXSparse && $(MAKE) purge )
	- ( cd RBio && $(MAKE) purge )
	- ( cd SPQR && $(MAKE) purge )
	- $(RM) MATLAB_Tools/*/*.mex* MATLAB_Tools/*/*/*.mex*
	- $(RM) MATLAB_Tools/*/*.o    MATLAB_Tools/*/*/*.o
	- $(RM) -r Example/build/*
	- ( cd GraphBLAS && $(MAKE) purge )
	- ( cd ParU && $(MAKE) purge )
	- ( cd LAGraph && $(MAKE) purge )
	- $(RM) -r include/* bin/* lib/*

clean: purge

# Run all demos
demos:
	- ( cd SuiteSparse_config && $(MAKE) demos )
	- ( cd AMD && $(MAKE) demos )
	- ( cd COLAMD && $(MAKE) demos )
	- ( cd SPEX && $(MAKE) demos )
	- ( cd Mongoose && $(MAKE) demos )
	- ( cd CAMD && $(MAKE) demos )
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
	- ( cd ParU && $(MAKE) demos )
	- ( cd LAGraph && $(MAKE) demos )

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
	( cd ParU && $(MAKE) docs )
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
	( cd LAGraph && $(MAKE) cov )

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
	( cd ParU && $(MAKE) debug )
	( cd RBio && $(MAKE) debug )
	( cd SPQR && $(MAKE) debug )
	( cd SPEX && $(MAKE) debug )
	( cd GraphBLAS && $(MAKE) cdebug )
	( cd LAGraph && $(MAKE) debug )

tests:
	( cd Mongoose && $(MAKE) test )
	( cd CHOLMOD && $(MAKE) test )
	( cd LAGraph && $(MAKE) test )

test: tests

