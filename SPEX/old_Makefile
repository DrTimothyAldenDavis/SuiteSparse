#-------------------------------------------------------------------------------
# Makefile for SPEX: SParse EXact linear algebra
#
# SPEX (c) 2019-2022 Christopher Lourenco, Timothy Davis, Erick Moreno-Centeno
#                    Jinhao Chen, All rights reserved.
#
#-------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/..)
export SUITESPARSE

default: go

include ../SuiteSparse_config/SuiteSparse_config.mk

# Compile the default rules for each package
go:
	( cd SPEX_Util && $(MAKE) )
	( cd SPEX_Left_LU && $(MAKE) )

# install all packages in /usr/local/lib and /usr/local/include
# (note that CSparse is not installed; CXSparse is installed instead)
install:
	( cd SPEX_Util && $(MAKE)  install )
	( cd SPEX_Left_LU && $(MAKE) install )
	$(CP) Doc/SPEX_UserGuide.pdf $(INSTALL_DOC)
	$(CP) README.md $(INSTALL_DOC)/SPEX_README.md
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_DOC)/SPEX_UserGuide.pdf
	chmod 644 $(INSTALL_DOC)/SPEX_README.md

# uninstall all packages
uninstall:
	( cd SPEX_Util && $(MAKE) uninstall )
	( cd SPEX_Left_LU && $(MAKE) uninstall )

# compile the libraries.
library: go

static: go

# test coverage:
cov:
	( cd SPEX_Left_LU && $(MAKE) cov )

# Remove all files not in the original distribution
purge:
	- ( cd SPEX_Util && $(MAKE) purge )
	- ( cd SPEX_Left_LU && $(MAKE) purge )

# Remove all files not in the original distribution, but keep the libraries
clean: purge

# Create the PDF documentation
docs:
	( cd Doc && $(MAKE) docs )

distclean: purge

