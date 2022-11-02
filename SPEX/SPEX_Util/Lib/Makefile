#-------------------------------------------------------------------------------
# SPEX_Util/Lib/Makefile
#
# SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later
#-------------------------------------------------------------------------------

# To just compile the library, do:
#       make
#
# To run a demo using the library
#       cd ../Demo ; make

LIBRARY = libspexutil
VERSION = 1.1.0
SO_VERSION = 1

default: library

SUITESPARSE ?= $(realpath $(CURDIR)/../../..)
include ../../../SuiteSparse_config/SuiteSparse_config.mk

# uncomment for extra error checking:
CFLAGS += -Wall -Wextra -Wpedantic -Werror

# SPEX_Util depends on SuiteSparse_config, AMD, COLAMD, M, GMP, and MPFR
LDLIBS += -lsuitesparseconfig -lamd -lcolamd -lm -lgmp -lmpfr

C = $(CC) $(CF) -I../Include -I../../../COLAMD/Include -I../../../AMD/Include -I../../../SuiteSparse_config

all: install

OBJ = \
    SPEX_matrix_div.o spex_create_mpq_array.o SPEX_free.o SPEX_realloc.o SPEX_matrix_mul.o  \
    spex_create_mpz_array.o SPEX_cumsum.o SPEX_malloc.o spex_sparse_collapse.o \
    SPEX_calloc.o SPEX_matrix_allocate.o  spex_sparse_realloc.o spex_cast_array.o spex_create_mpfr_array.o \
    spex_expand_double_array.o SPEX_gmp.o SPEX_matrix_copy.o SPEX_matrix_check.o spex_cast_matrix.o \
    spex_expand_mpfr_array.o SPEX_initialize.o SPEX_matrix_free.o SPEX_check_solution.o SPEX_finalize.o  \
    spex_expand_mpq_array.o SPEX_initialize_expert.o SPEX_matrix_nnz.o SPEX_create_default_options.o  \
    SPEX_LU_analysis_free.o SPEX_LU_analyze.o

$(OBJ): ../Include/SPEX_Util.h ../Source/spex_util_internal.h Makefile

%.o: ../Source/%.c ../Include/SPEX_Util.h ../Source/spex_util_internal.h
	$(C) $(I) -c $<

#-------------------------------------------------------------------------------
# Create the bin
#-------------------------------------------------------------------------------

library:
	$(MAKE) install INSTALL=$(SUITESPARSE)

static: $(AR_TARGET)

$(AR_TARGET): $(OBJ)
	$(ARCHIVE) $@ $^
	- $(RANLIB) $(AR_TARGET)

#-------------------------------------------------------------------------------
# install (shared C version only)
#-------------------------------------------------------------------------------

# install SPEX_Util
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJ)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CC) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../Include/SPEX_Util.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_INCLUDE)/SPEX_Util.h

# uninstall SPEX_Util
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_INCLUDE)/SPEX_Util.h

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

