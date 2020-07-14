#-------------------------------------------------------------------------------
# SLIP_LU/Lib/Makefile
#-------------------------------------------------------------------------------

# To just compile the library, do:
#       make
#
# To run a demo using the library
#       cd ../Demo ; make

LIBRARY = libsliplu
VERSION = 1.0.2
SO_VERSION = 1

default: library

include ../../SuiteSparse_config/SuiteSparse_config.mk

# uncomment for extra error checking:
# CFLAGS += -Wall -Wextra -Wpedantic -Werror

# SLIP_LU depends on SuiteSparse_config, AMD, COLAMD, M, GMP, and MPFR
LDLIBS += -lsuitesparseconfig -lamd -lcolamd -lm -lgmp -lmpfr

C = $(CC) $(CF) -I../Include -I../../COLAMD/Include -I../../AMD/Include -I../../SuiteSparse_config

all: install

OBJ = \
    slip_matrix_div.o slip_create_mpq_array.o SLIP_free.o SLIP_LU_factorize.o \
    SLIP_realloc.o slip_matrix_mul.o  slip_get_largest_pivot.o  \
    slip_ref_triangular_solve.o SLIP_backslash.o slip_create_mpz_array.o slip_get_nonzero_pivot.o SLIP_LU_solve.o \
    slip_back_sub.o slip_cumsum.o slip_get_pivot.o SLIP_malloc.o slip_sparse_collapse.o \
    SLIP_calloc.o slip_dfs.o slip_get_smallest_pivot.o SLIP_matrix_allocate.o  slip_sparse_realloc.o \
    slip_cast_array.o slip_expand_double_array.o SLIP_gmp.o SLIP_matrix_copy.o SLIP_matrix_check.o \
    slip_cast_matrix.o slip_expand_mpfr_array.o SLIP_initialize.o SLIP_matrix_free.o slip_check_solution.o \
    slip_expand_mpq_array.o SLIP_initialize_expert.o SLIP_matrix_nnz.o SLIP_create_default_options.o  \
    SLIP_finalize.o SLIP_LU_analysis_free.o slip_permute_x.o slip_permute_b.o \
    slip_create_mpfr_array.o \
    slip_forward_sub.o SLIP_LU_analyze.o slip_reach.o


$(OBJ): ../Include/SLIP_LU.h ../Source/slip_internal.h Makefile

%.o: ../Source/%.c ../Include/SLIP_LU.h ../Source/slip_internal.h
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

# install SLIP_LU
install: $(AR_TARGET) $(INSTALL_LIB)/$(SO_TARGET)

$(INSTALL_LIB)/$(SO_TARGET): $(OBJ)
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	$(CC) $(SO_OPTS) $^ -o $@ $(LDLIBS)
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_PLAIN) )
	( cd $(INSTALL_LIB) ; ln -sf $(SO_TARGET) $(SO_MAIN) )
	$(CP) ../Include/SLIP_LU.h $(INSTALL_INCLUDE)
	$(CP) ../Doc/SLIP_LU_UserGuide.pdf $(INSTALL_DOC)
	$(CP) ../README.md $(INSTALL_DOC)/SLIP_LU_README.md
	chmod 755 $(INSTALL_LIB)/$(SO_TARGET)
	chmod 644 $(INSTALL_INCLUDE)/SLIP_LU.h
	chmod 644 $(INSTALL_DOC)/SLIP_LU_UserGuide.pdf
	chmod 644 $(INSTALL_DOC)/SLIP_LU_README.md

# uninstall SLIP_LU
uninstall:
	$(RM) $(INSTALL_LIB)/$(SO_TARGET)
	$(RM) $(INSTALL_LIB)/$(SO_PLAIN)
	$(RM) $(INSTALL_LIB)/$(SO_MAIN)
	$(RM) $(INSTALL_INCLUDE)/SLIP_LU.h
	$(RM) $(INSTALL_DOC)/SLIP_LU_UserGuide.pdf
	$(RM) $(INSTALL_DOC)/SLIP_LU_README.md

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) -r $(PURGE)

