#-------------------------------------------------------------------------------
# Makefile for SuiteSparse/TestConfig packages
#-------------------------------------------------------------------------------

# SuiteSparse/TestConfig, Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# precede this test with "make local ; make install" in the top-level
# SuiteSparse directory.

CONFIG = cmake -DCMAKE_PREFIX_PATH="../../lib/cmake" ..
BUILD  = cmake --build . --config Release
V = 
# V = valgrind --leak-check=full --show-leak-kinds=all

test:
	(cd AMD/build                && $(CONFIG) && $(BUILD) && $(V) ./amd_demo && ./amd_demo_static )
	(cd BTF/build                && $(CONFIG) && $(BUILD) && $(V) ./btf_demo && ./btf_demo_static )
	(cd CAMD/build               && $(CONFIG) && $(BUILD) && $(V) ./camd_demo && ./camd_demo_static )
	(cd CCOLAMD/build            && $(CONFIG) && $(BUILD) && $(V) ./ccolamd_demo && ./ccolamd_demo_static )
	(cd CHOLMOD/build            && $(CONFIG) && $(BUILD) && $(V) ./cholmod_demo && ./cholmod_demo_static )
	(cd COLAMD/build             && $(CONFIG) && $(BUILD) && $(V) ./colamd_demo && ./colamd_demo_static )
	(cd CXSparse/build           && $(CONFIG) && $(BUILD) && $(V) ./cxsparse_demo && ./cxsparse_demo_static )
	(cd GraphBLAS/build          && $(CONFIG) && $(BUILD) && $(V) ./graphblas_demo && ./graphblas_demo_static )
	(cd KLU/build                && $(CONFIG) && $(BUILD) && $(V) ./klu_demo && ./klu_demo_static )
	(cd LAGraph/build            && $(CONFIG) && $(BUILD) && $(V) ./lagraph_demo && ./lagraph_demo_static )
	(cd LDL/build                && $(CONFIG) && $(BUILD) && $(V) ./ldl_demo && ./ldl_demo_static )
	(cd Mongoose/build           && $(CONFIG) && $(BUILD) && $(V) ./mongoose_demo && ./mongoose_demo_static )
	(cd ParU/build               && $(CONFIG) && $(BUILD) && $(V) ./paru_demo && ./paru_demo_static )
	(cd RBio/build               && $(CONFIG) && $(BUILD) && $(V) ./rbio_demo && ./rbio_demo_static )
	(cd SPEX/build               && $(CONFIG) && $(BUILD) && $(V) ./spex_demo && ./spex_demo_static )
	(cd SPQR/build               && $(CONFIG) && $(BUILD) && $(V) ./spqr_demo && ./spqr_demo_static )
	(cd UMFPACK/build            && $(CONFIG) && $(BUILD) && $(V) ./umfpack_demo && ./umfpack_demo_static )
	(cd SuiteSparse_config/build && $(CONFIG) && $(BUILD) && $(V) ./config_demo && ./config_demo_static )

# Remove all files not in the original distribution
distclean: purge

purge:
	- $(RM) -r */build/*

clean: purge

