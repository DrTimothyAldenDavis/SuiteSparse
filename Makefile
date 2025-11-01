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

# Note that CSparse is compiled but not installed by "make install";
# CXSparse is installed instead.

# default: "make install" will install all libraries in /usr/local/lib
# and include files in /usr/local/include.  Not installed in SuiteSparse/lib.
default: library

# compile; "sudo make install" will install only in /usr/local
# (or whatever your CMAKE_INSTALL_PREFIX is)
library:
	( cd SuiteSparse_config && $(MAKE) )
	( cd AMD && $(MAKE) )
	( cd COLAMD && $(MAKE) )
	( cd CAMD && $(MAKE) )
	( cd CCOLAMD && $(MAKE) )
	( cd CHOLMOD && $(MAKE) )
	( cd UMFPACK && $(MAKE) )
	( cd ParU && $(MAKE) )

# compile; "make install" only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd SuiteSparse_config && $(MAKE) local )
	( cd AMD && $(MAKE) local )
	( cd COLAMD && $(MAKE) local )
	( cd CAMD && $(MAKE) local )
	( cd CCOLAMD && $(MAKE) local )
	( cd CHOLMOD && $(MAKE) local )
	( cd UMFPACK && $(MAKE) local )
	( cd ParU && $(MAKE) local )

# compile; "sudo make install" will install only in /usr/local
# (or whatever your CMAKE_INSTALL_PREFIX is)
global:
	( cd SuiteSparse_config && $(MAKE) global )
	( cd AMD && $(MAKE) global )
	( cd COLAMD && $(MAKE) global )
	( cd CAMD && $(MAKE) global )
	( cd CCOLAMD && $(MAKE) global )
	( cd CHOLMOD && $(MAKE) global )
	( cd UMFPACK && $(MAKE) global )
	( cd ParU && $(MAKE) global )

# install all packages.  Location depends on prior "make", "make global" etc
install:
	( cd SuiteSparse_config && $(MAKE) install )
	( cd AMD && $(MAKE) install )
	( cd COLAMD && $(MAKE) install )
	( cd CAMD && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) install )
	( cd UMFPACK && $(MAKE) install )
	( cd ParU && $(MAKE) install )

# uninstall all packages
uninstall:
	( cd SuiteSparse_config && $(MAKE) uninstall )
	( cd AMD && $(MAKE) uninstall )
	( cd COLAMD && $(MAKE) uninstall )
	( cd CAMD && $(MAKE) uninstall )
	( cd CCOLAMD && $(MAKE) uninstall )
	( cd ParU && $(MAKE) uninstall )
	( cd UMFPACK && $(MAKE) uninstall )
	( cd CHOLMOD && $(MAKE) uninstall )

# Remove all files not in the original distribution
distclean: purge

# Remove all files not in the original distribution
purge:
	- ( cd SuiteSparse_config && $(MAKE) purge )
	- ( cd AMD && $(MAKE) purge )
	- ( cd COLAMD && $(MAKE) purge )
	- ( cd CAMD && $(MAKE) purge )
	- ( cd CCOLAMD && $(MAKE) purge )
	- ( cd UMFPACK && $(MAKE) purge )
	- ( cd CHOLMOD && $(MAKE) purge )
	- ( cd ParU && $(MAKE) purge )
	- $(RM) -r include/* bin/* lib/* build/*

clean: purge

# Run all demos
demos:
	- ( cd SuiteSparse_config && $(MAKE) demos )
	- ( cd AMD && $(MAKE) demos )
	- ( cd COLAMD && $(MAKE) demos )
	- ( cd CAMD && $(MAKE) demos )
	- ( cd CCOLAMD && $(MAKE) demos )
	- ( cd UMFPACK && $(MAKE) demos )
	- ( cd CHOLMOD && $(MAKE) demos )
	- ( cd ParU && $(MAKE) demos )

# Create the PDF documentation
docs:
	( cd AMD && $(MAKE) docs )
	( cd CAMD && $(MAKE) docs )
	( cd UMFPACK && $(MAKE) docs )
	( cd CHOLMOD && $(MAKE) docs )
	( cd ParU && $(MAKE) docs )

# statement coverage (Linux only); this requires a lot of time.
cov: local install
	( cd CHOLMOD && $(MAKE) cov )
	( cd UMFPACK && $(MAKE) cov )

debug:
	( cd SuiteSparse_config && $(MAKE) debug )
	( cd AMD && $(MAKE) debug )
	( cd CAMD && $(MAKE) debug )
	( cd CCOLAMD && $(MAKE) debug )
	( cd COLAMD && $(MAKE) debug )
	( cd CHOLMOD && $(MAKE) debug )
	( cd UMFPACK && $(MAKE) debug )
	( cd ParU && $(MAKE) debug )

tests:
	( cd CHOLMOD && $(MAKE) test )

test: tests

