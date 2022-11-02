#-------------------------------------------------------------------------------
# SPEX_Left_LU/Lib/Makefile
#
# SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later
#-------------------------------------------------------------------------------

# To just compile the library, do:
#       make
#
# To run a demo using the library
#       cd ../Demo ; make

LIBRARY = libspexleftlu
VERSION = 1.0.1
SO_VERSION = 1

default: library

SUITESPARSE ?= $(realpath $(CURDIR)/../../..)
include ../../../SuiteSparse_config/SuiteSparse_config.mk

# uncomment for extra error checking:
CFLAGS += -Wall -Wextra -Wpedantic -Werror

# SPEX_Left LU depends on SuiteSparse_config, AMD, COLAMD, M, GMP, and MPFR
LDLIBS += -lspexutil -lsuitesparseconfig -lamd -lcolamd -lm -lgmp -lmpfr

C = $(CC) $(CF) -I../Include -I../../../COLAMD/Include -I../../../AMD/Include -I../../../SuiteSparse_config -I../../../include/ -I../../SPEX_Util/Include -I../../SPEX_Util/Source

all: install

OBJ = \
     spex_left_lu_back_sub.o spex_left_lu_forward_sub.o \
     spex_left_lu_get_pivot.o \
     spex_left_lu_permute_b.o spex_left_lu_ref_triangular_solve.o \
     spex_left_lu_dfs.o \
     spex_left_lu_get_largest_pivot.o  spex_left_lu_get_smallest_pivot.o  \
     spex_left_lu_permute_x.o \
     SPEX_Left_LU_solve.o SPEX_Left_LU_backslash.o SPEX_Left_LU_factorize.o \
     spex_left_lu_get_nonzero_pivot.o \
     spex_left_lu_reach.o


$(OBJ): ../Include/SPEX_Left_LU.h ../Source/spex_left_lu_internal.h Makefile

%.o: ../Source/%.c ../Include/SPEX_Left_LU.h ../Source/spex_left_lu_internal.h
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

# install SPEX_Left_LU
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJ)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CC) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../Include/SPEX_Left_LU.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_INCLUDE)/SPEX_Left_LU.h

# uninstall SPEX_Left_LU
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_INCLUDE)/SPEX_Left_LU.h

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

