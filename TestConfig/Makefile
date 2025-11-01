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
	(cd CAMD/build               && $(CONFIG) && $(BUILD) && $(V) ./camd_demo && ./camd_demo_static )
	(cd CCOLAMD/build            && $(CONFIG) && $(BUILD) && $(V) ./ccolamd_demo && ./ccolamd_demo_static )
	(cd CHOLMOD/build            && $(CONFIG) && $(BUILD) && $(V) ./cholmod_demo && ./cholmod_demo_static )
	(cd COLAMD/build             && $(CONFIG) && $(BUILD) && $(V) ./colamd_demo && ./colamd_demo_static )
	(cd ParU/build               && $(CONFIG) && $(BUILD) && $(V) ./paru_demo && ./paru_demo_static )
	(cd UMFPACK/build            && $(CONFIG) && $(BUILD) && $(V) ./umfpack_demo && ./umfpack_demo_static )
	(cd SuiteSparse_config/build && $(CONFIG) && $(BUILD) && $(V) ./config_demo && ./config_demo_static )

# Remove all files not in the original distribution
distclean: purge

purge:
	- $(RM) -r */build/*

clean: purge

