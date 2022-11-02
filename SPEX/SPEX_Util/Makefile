#------------------------------------------------------------------------------
# SPEX_Util/Makefile: compile, install, and run SPEX_Util demo and tests
#------------------------------------------------------------------------------

# SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later
#------------------------------------------------------------------------------

SUITESPARSE ?= $(realpath $(CURDIR)/../..)
export SUITESPARSE

# default: compile the dynamic library, and run the demos
default: C

include ../../SuiteSparse_config/SuiteSparse_config.mk

# compile the dynamic library, and then run the demos and statement coverage
demos: all

# compile the dynamic library only, and then run the demos
C:
	( cd Lib ; $(MAKE) )
	#( cd Demo ; $(MAKE) )

# compile the dynamic library, and then run the demos and statement coverage
all: C cov

# compile the dynamic library only
library:
	( cd Lib ; $(MAKE) )

# compile the static library only
static:
	( cd Lib ; $(MAKE) static )

# statement coverage test
cov:
	- ( cd Tcov ; $(MAKE) )

# remove files not in the distribution, but keep compiled libraries
clean:
	( cd Lib ; $(MAKE) clean )

# remove all files not in the distribution
purge:
	( cd Lib ; $(MAKE) purge )

distclean: purge

# install the library
install: library
	( cd Lib ; $(MAKE) install )

# uninstall the library
uninstall:
	( cd Lib ; $(MAKE) uninstall )

