#-------------------------------------------------------------------------------
# SuiteSparse/CMakeLists.txt:  root CMake build rules
#-------------------------------------------------------------------------------

# Copyright (c) 2023-2024, Timothy A. Davis, All Rights Reserved.
# Just this particular file is under the Apache-2.0 license; each package has
# its own license.
# SPDX-License-Identifier: Apache-2.0

# This file and most packages in SuiteSparse require cmake 3.22 or later.  Some
# packages can be built as stand-alone packages with their own CMakeLists.txt
# files, with older versions of cmake (GraphBLAS, LAGraph, and CSparse):
#
#   GraphBLAS and LAGraph: 3.20
#   CSparse: 3.13
#   GraphBLAS jitifyer (for end user JIT kernels): 3.13
#
# Other CMakeLists.txt files inside SuiteSparse are from dependent packages
# (LAGraph/deps/json_h, GraphBLAS/cpu_features, and CHOLMOD/SuiteSparse_metis
# which is a slightly revised copy of METIS 5.0.1) but none of those
# CMakeLists.txt files are used to build any package in SuiteSparse.
cmake_minimum_required ( VERSION 3.22 )

project ( "SuiteSparse" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/SuiteSparse_config/cmake_modules )

#-------------------------------------------------------------------------------
# build options
#-------------------------------------------------------------------------------

# lower-case list of all projects that can be built by this root CMake file
set ( SUITESPARSE_ALL_PROJECTS
    "suitesparse_config;mongoose;amd;btf;camd;ccolamd;colamd;cholmod;cxsparse;ldl;klu;umfpack;paru;rbio;spqr;spex;graphblas;lagraph" )

# lower-case list of extra projects that can be built by this root CMake file
set ( SUITESPARSE_EXTRA_PROJECTS
    "csparse" )

# lower-case list of known projects that can be built by this root CMake file
set ( SUITESPARSE_KNOWN_PROJECTS "${SUITESPARSE_ALL_PROJECTS};${SUITESPARSE_EXTRA_PROJECTS}" )

set ( SUITESPARSE_ENABLE_PROJECTS "all" CACHE STRING
    "Semicolon-separated list of SuiteSparse projects to be built (${SUITESPARSE_KNOWN_PROJECTS}, or \"all\")" )

# expand "all" early on
if ( "all" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    set ( SUITESPARSE_ENABLE_PROJECTS "${SUITESPARSE_ENABLE_PROJECTS};${SUITESPARSE_ALL_PROJECTS}" )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "all" )
    list ( REMOVE_DUPLICATES  SUITESPARSE_ENABLE_PROJECTS )
endif ( )

# check for unknown projects in list
foreach ( proj ${SUITESPARSE_ENABLE_PROJECTS} )
  if ( NOT "${proj}" IN_LIST SUITESPARSE_KNOWN_PROJECTS )
     message ( FATAL_ERROR "${proj} is not a known project: ${SUITESPARSE_KNOWN_PROJECTS}." )
  endif ( )
endforeach ( )

# CHOLMOD options affecting dependencies
option ( CHOLMOD_CAMD "ON (default): use CAMD/CCOLAMD.  OFF: do not use CAMD/CCOLAMD" ON )

# KLU options affecting dependencies
option ( KLU_USE_CHOLMOD "ON (default): use CHOLMOD in KLU.  OFF: do not use CHOLMOD in KLU" ON )

# UMFPACK options affecting dependencies
option ( UMFPACK_USE_CHOLMOD "ON (default): use CHOLMOD in UMFPACK.  OFF: do not use CHOLMOD in UMFPACK" ON )

# overwrite BUILD_STATIC_LIBS specifically for GraphBLAS because building the
# library takes a long time
option ( GRAPHBLAS_BUILD_STATIC_LIBS "OFF (default): Do not build static libraries for GraphBLAS project.  ON: Use same value of BUILD_STATIC_LIBS for GraphBLAS like in the other projects" OFF )

# options to build with libraries installed on the system instead of building
# dependencies automatically
option ( SUITESPARSE_USE_SYSTEM_BTF "ON: use BTF libraries installed on the build system.  OFF (default): Automatically build BTF as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_CHOLMOD "ON: use CHOLMOD libraries installed on the build system.  OFF (default): Automatically build CHOLMOD as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_AMD "ON: use AMD libraries installed on the build system.  OFF (default): Automatically build AMD as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_COLAMD "ON: use COLAMD libraries installed on the build system.  OFF (default): Automatically build COLAMD as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_CAMD "ON: use CAMD libraries installed on the build system.  OFF (default): Automatically build CAMD as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_CCOLAMD "ON: use CCOLAMD libraries installed on the build system.  OFF (default): Automatically build CCOLAMD as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_GRAPHBLAS "ON: use GraphBLAS libraries installed on the build system.  OFF (default): Automatically build GraphBLAS as dependency if needed." OFF )
option ( SUITESPARSE_USE_SYSTEM_SUITESPARSE_CONFIG "ON: use SuiteSparse_config libraries installed on the build system.  OFF (default): Automatically build SuiteSparse_config as dependency if needed." OFF )

#-------------------------------------------------------------------------------
# global variables
#-------------------------------------------------------------------------------

# Set to indicate that we are building from a root CMake file.
# That will change the directory layout and (imported) target names (namespace!)
# during the build process.
set ( SUITESPARSE_ROOT_CMAKELISTS ON )

#-------------------------------------------------------------------------------
# common SuiteSparse modules
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# check/add project dependencies
#-------------------------------------------------------------------------------

if ( SUITESPARSE_USE_SYSTEM_GRAPHBLAS )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "graphblas" )
    find_package ( GraphBLAS 9.0.1 REQUIRED )
else ( )
    if ( "lagraph" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # LAGraph requires GraphBLAS.
        if ( NOT "graphblas" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"graphblas\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "graphblas" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_BTF )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "btf" )
    find_package ( BTF 2.3.1 REQUIRED )
else ( )
    if ( "klu" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # KLU requires BTF.
        if ( NOT "btf" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"btf\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "btf" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_CHOLMOD )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "cholmod" )
    find_package ( CHOLMOD 5.2.0 REQUIRED )
else ( )
    if ( ( KLU_USE_CHOLMOD AND "klu" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            OR ( UMFPACK_USE_CHOLMOD AND "umfpack" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            OR "spqr" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "paru" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # SPQR and ParU both require CHOLMOD.  KLU and UMFPACK can optionally
        # use CHOLMOD.  Add CHOLMOD to the list of projects, if it has been
        # requested by SPQR, ParU, KLU, or UMFPACK, if not already there.
        if ( NOT "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"cholmod\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "cholmod" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_AMD )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "amd" )
    find_package ( AMD 3.3.1 REQUIRED )
else ( )
    if ( "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "ldl" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "umfpack" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "spex" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # CHOLMOD, LDL, UMFPACK, and SPEX require AMD.
        if ( NOT "amd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"amd\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "amd" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_COLAMD )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "colamd" )
    find_package ( COLAMD 3.3.2 REQUIRED )
else ( )
    if ( "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "spex" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # CHOLMOD and SPEX require COLAMD.
        if ( NOT "colamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"colamd\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "colamd" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_CAMD )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "camd" )
    find_package ( CAMD 3.3.1 REQUIRED )
else ( )
    if ( CHOLMOD_CAMD AND "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # CHOLMOD can optionally use CAMD.
        if ( NOT SUITESPARSE_USE_SYSTEM_CAMD AND NOT "camd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"camd\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "camd" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_CCOLAMD )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "ccolamd" )
    find_package ( CCOLAMD 3.3.2 REQUIRED )
else ( )
    if ( CHOLMOD_CAMD AND "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # CHOLMOD can optionally use CCOLAMD.
        if ( NOT "ccolamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"ccolamd\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "ccolamd" )
        endif ( )
    endif ( )
endif ( )

if ( SUITESPARSE_USE_SYSTEM_SUITESPARSE_CONFIG )
    list ( REMOVE_ITEM SUITESPARSE_ENABLE_PROJECTS "suitesparse_config" )
    find_package ( SuiteSparse_config 7.6.0 REQUIRED )
else ( )
    if ( "mongoose" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "amd" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "btf" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "camd" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "ccolamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "colamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "cxsparse" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "ldl" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "klu" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "umfpack" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "paru" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "rbio" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "spqr" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "spex" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        # All but CSparse, GraphBLAS, and LAGraph require SuiteSparse_config.
        if ( NOT "suitesparse_config" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
            message ( STATUS "Adding \"suitesparse_config\" to the list of built targets." )
            list ( APPEND SUITESPARSE_ENABLE_PROJECTS "suitesparse_config" )
        endif ( )
    endif ( )
endif ( )


if ( CMAKE_VERSION VERSION_LESS 3.24 )
    # work around missing GLOBAL option of find_package in older CMake versions
    # If SuiteSparse is included as a sub-project in other projects, they might
    # need to manually import the OpenMP targets for older CMake versions, too.
    if ( "suitesparse_config" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "graphblas" IN_LIST SUITESPARSE_ENABLE_PROJECTS
            OR "paru" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        find_package ( OpenMP COMPONENTS C )
    endif ( )
    if ( "paru" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        find_package ( OpenMP COMPONENTS CXX )
    endif ( )
endif ( )


#-------------------------------------------------------------------------------
# include selected projects
#-------------------------------------------------------------------------------

if ( "suitesparse_config" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "SuiteSparse_config" )
    if ( TARGET SuiteSparseConfig )
        add_library ( SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig )
    else ( )
        add_library ( SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig_static )
    endif ( )
    if ( TARGET SuiteSparseConfig_static )
        add_library ( SuiteSparse::SuiteSparseConfig_static ALIAS SuiteSparseConfig_static )
    endif ( )
endif ( )

if ( "mongoose" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "Mongoose" )
    if ( TARGET Mongoose )
        add_library ( SuiteSparse::Mongoose ALIAS Mongoose )
    else ( )
        add_library ( SuiteSparse::Mongoose ALIAS Mongoose_static )
    endif ( )
     if ( TARGET Mongoose_static )
        add_library ( SuiteSparse::Mongoose_static ALIAS Mongoose_static )
    endif ( )
endif ( )

if ( "amd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "AMD" )
    if ( TARGET AMD )
        add_library ( SuiteSparse::AMD ALIAS AMD )
    else ( )
        add_library ( SuiteSparse::AMD ALIAS AMD_static )
    endif ( )
    if ( TARGET AMD_static )
        add_library ( SuiteSparse::AMD_static ALIAS AMD_static )
    endif ( )
endif ( )

if ( "btf" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "BTF" )
    if ( TARGET BTF )
        add_library ( SuiteSparse::BTF ALIAS BTF )
    else ( )
        add_library ( SuiteSparse::BTF ALIAS BTF_static )
    endif ( )
    if ( TARGET BTF_static )
        add_library ( SuiteSparse::BTF_static ALIAS BTF_static )
    endif ( )
endif ( )

if ( "camd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CAMD" )
    if ( TARGET CAMD )
        add_library ( SuiteSparse::CAMD ALIAS CAMD )
    else ( )
        add_library ( SuiteSparse::CAMD ALIAS CAMD_static )
    endif ( )
    if ( TARGET CAMD_static )
        add_library ( SuiteSparse::CAMD_static ALIAS CAMD_static )
    endif ( )
endif ( )

if ( "ccolamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CCOLAMD" )
    if ( TARGET CCOLAMD )
        add_library ( SuiteSparse::CCOLAMD ALIAS CCOLAMD )
    else ( )
        add_library ( SuiteSparse::CCOLAMD ALIAS CCOLAMD_static )
    endif ( )
    if ( TARGET CCOLAMD_static )
        add_library ( SuiteSparse::CCOLAMD_static ALIAS CCOLAMD_static )
    endif ( )
endif ( )

if ( "colamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "COLAMD" )
    if ( TARGET COLAMD )
        add_library ( SuiteSparse::COLAMD ALIAS COLAMD )
    else ( )
        add_library ( SuiteSparse::COLAMD ALIAS COLAMD_static )
    endif ( )
    if ( TARGET COLAMD_static )
        add_library ( SuiteSparse::COLAMD_static ALIAS COLAMD_static )
    endif ( )
endif ( )

if ( "cholmod" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CHOLMOD" )
    if ( TARGET CHOLMOD )
        add_library ( SuiteSparse::CHOLMOD ALIAS CHOLMOD )
    else ( )
        add_library ( SuiteSparse::CHOLMOD ALIAS CHOLMOD_static )
    endif ( )
    if ( TARGET CHOLMOD_static )
        add_library ( SuiteSparse::CHOLMOD_static ALIAS CHOLMOD_static )
    endif ( )
    if ( TARGET CHOLMOD_CUDA )
        add_library ( SuiteSparse::CHOLMOD_CUDA ALIAS CHOLMOD_CUDA )
    elseif ( TARGET CHOLMOD_CUDA_static )
        add_library ( SuiteSparse::CHOLMOD_CUDA ALIAS CHOLMOD_CUDA_static )
    endif ( )
    if ( TARGET CHOLMOD_CUDA_static )
        add_library ( SuiteSparse::CHOLMOD_CUDA_static ALIAS CHOLMOD_CUDA_static )
    endif ( )
endif ( )

if ( "cxsparse" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CXSparse" )
    if ( TARGET CXSparse )
        add_library ( SuiteSparse::CXSparse ALIAS CXSparse )
    else ( )
        add_library ( SuiteSparse::CXSparse ALIAS CXSparse_static )
    endif ( )
    if ( TARGET CXSparse_static )
        add_library ( SuiteSparse::CXSparse_static ALIAS CXSparse_static )
    endif ( )
endif ( )

if ( "ldl" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "LDL" )
    if ( TARGET LDL )
        add_library ( SuiteSparse::LDL ALIAS LDL )
    else ( )
        add_library ( SuiteSparse::LDL ALIAS LDL_static )
    endif ( )
    if ( TARGET LDL_static )
        add_library ( SuiteSparse::LDL_static ALIAS LDL_static )
    endif ( )
endif ( )

if ( "klu" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "KLU" )
    if ( TARGET KLU )
        add_library ( SuiteSparse::KLU ALIAS KLU )
    else ( )
        add_library ( SuiteSparse::KLU ALIAS KLU_static )
    endif ( )
    if ( TARGET KLU_static )
        add_library ( SuiteSparse::KLU_static ALIAS KLU_static )
    endif ( )
endif ( )

if ( "umfpack" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "UMFPACK" )
    if ( TARGET UMFPACK )
        add_library ( SuiteSparse::UMFPACK ALIAS UMFPACK )
    else ( )
        add_library ( SuiteSparse::UMFPACK ALIAS UMFPACK_static )
    endif ( )
    if ( TARGET UMFPACK_static )
        add_library ( SuiteSparse::UMFPACK_static ALIAS UMFPACK_static )
    endif ( )
endif ( )

if ( "paru" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "ParU" )
    if ( TARGET ParU )
        add_library ( SuiteSparse::ParU ALIAS ParU )
    else ( )
        add_library ( SuiteSparse::ParU ALIAS ParU_static )
    endif ( )
    if ( TARGET ParU_static )
        add_library ( SuiteSparse::ParU_static ALIAS ParU_static )
    endif ( )
endif ( )

if ( "rbio" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "RBio" )
    if ( TARGET RBio )
        add_library ( SuiteSparse::RBio ALIAS RBio )
    else ( )
        add_library ( SuiteSparse::RBio ALIAS RBio_static )
    endif ( )
    if ( TARGET RBio_static )
        add_library ( SuiteSparse::RBio_static ALIAS RBio_static )
    endif ( )
endif ( )

if ( "spqr" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "SPQR" )
    if ( TARGET SPQR )
        add_library ( SuiteSparse::SPQR ALIAS SPQR )
    else ( )
        add_library ( SuiteSparse::SPQR ALIAS SPQR_static )
    endif ( )
    if ( TARGET SPQR_static )
        add_library ( SuiteSparse::SPQR_static ALIAS SPQR_static )
    endif ( )
endif ( )

if ( "spex" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "SPEX" )
    if ( TARGET SPEX )
        add_library ( SuiteSparse::SPEX ALIAS SPEX )
    else ( )
        add_library ( SuiteSparse::SPEX ALIAS SPEX_static )
    endif ( )
    if ( TARGET SPEX_static )
        add_library ( SuiteSparse::SPEX_static ALIAS SPEX_static )
    endif ( )
endif ( )

if ( "graphblas" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "GraphBLAS" )
    if ( TARGET GraphBLAS )
        add_library ( SuiteSparse::GraphBLAS ALIAS GraphBLAS )
    else ( )
        add_library ( SuiteSparse::GraphBLAS ALIAS GraphBLAS_static )
    endif ( )
    if ( TARGET GraphBLAS_static )
        add_library ( SuiteSparse::GraphBLAS_static ALIAS GraphBLAS_static )
    endif ( )
endif ( )

if ( "lagraph" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "LAGraph" )
    if ( TARGET LAGraph )
        add_library ( SuiteSparse::LAGraph ALIAS LAGraph )
    else ( )
        add_library ( SuiteSparse::LAGraph ALIAS LAGraph_static )
    endif ( )
    if ( TARGET LAGraph_static )
        add_library ( SuiteSparse::LAGraph_static ALIAS LAGraph_static )
    endif ( )
endif ( )

if ( "csparse" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CSparse" )
    if ( TARGET CSparse )
        add_library ( SuiteSparse::CSparse ALIAS CSparse )
    else ( )
        add_library ( SuiteSparse::CSparse ALIAS CSparse_static )
    endif ( )
    if ( TARGET CSparse_static )
        add_library ( SuiteSparse::CSparse_static ALIAS CSparse_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

#-------------------------------------------------------------------------------
# enable testing facilities
#-------------------------------------------------------------------------------

# Currently, only LAGraph, Mongoose, and CHOLMOD have ctests.

# FIXME: convert more of the existing demos to ctests.

# Most packages have a ./Tcov folder with a full statement coverage test,
# but these are not imported into cmake yet.

# Most packages also have a ./Demo folder, with shorter examples.  These would
# be nice to add as quick ctests.

# CHOLMOD/Tcov takes about 20 minutes to run.  It is also a full coverage
# test of AMD, CAMD, COLAMD, and CCOLAMD, however.  The current CHOLMOD
# ctest is based on a few ./Demo programs.  It's fast but not a full coverate
# test.

# The CSparse/CXSparse Tcov tests are very fast and would be good candidates to
# add.

include ( CTest )

#-------------------------------------------------------------------------------
# rule to remove all files in build directory
#-------------------------------------------------------------------------------

file ( GLOB SUITESPARSE_BUILT_FILES ${PROJECT_BINARY_DIR}/* )
file ( REAL_PATH ${PROJECT_SOURCE_DIR} _real_project_source_dir )
file ( REAL_PATH ${PROJECT_BINARY_DIR} _real_project_binary_dir )
if ( _real_project_source_dir STREQUAL _real_project_binary_dir )
    add_custom_target ( purge
        COMMENT "The target 'purge' is a no-op for in-tree builds.  Consider building out of the source tree." )
else ( )
    add_custom_target ( purge
        COMMAND ${CMAKE_COMMAND} -E echo "Removing files..."
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${SUITESPARSE_BUILT_FILES}
        COMMENT "Purge all files in the build tree" )
endif ( )
