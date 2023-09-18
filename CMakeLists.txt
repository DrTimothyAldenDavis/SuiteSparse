#-------------------------------------------------------------------------------
# SuiteSparse/CMakeLists.txt:  root CMake build rules
#-------------------------------------------------------------------------------

# Copyright (c) 2023, Timothy A. Davis.
# All Rights Reserved.

# The actually required minimum CMake version might be higher depending on the
# selected SuiteSparse projects.
cmake_minimum_required ( VERSION 3.20 )

project ( "SuiteSparse" )

#-------------------------------------------------------------------------------
# build options
#-------------------------------------------------------------------------------

# lower-case list of all projects that can be built by this root CMake file
set ( SUITESPARSE_KNOWN_PROJECTS
    "suitesparse_config;mongoose;amd;btf;camd;ccolamd" )

set ( SUITESPARSE_ENABLE_PROJECTS "all" CACHE STRING
    "Semicolon-separated list of SuiteSparse projects to be built (${SUITESPARSE_KNOWN_PROJECTS}, or \"all\")" )

# expand "all" early on
if ( SUITESPARSE_ENABLE_PROJECTS STREQUAL "all" )
    set ( SUITESPARSE_ENABLE_PROJECTS ${SUITESPARSE_KNOWN_PROJECTS} )
endif ( )

# check for unknown projects in list
foreach ( proj ${SUITESPARSE_ENABLE_PROJECTS} )
  if ( NOT "${proj}" IN_LIST SUITESPARSE_KNOWN_PROJECTS )
     message ( FATAL_ERROR "${proj} is not a known project: ${SUITESPARSE_KNOWN_PROJECTS}." )
  endif ( )
endforeach ( )

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

if ( "mongoose" IN_LIST SUITESPARSE_ENABLE_PROJECTS
        OR "amd" IN_LIST SUITESPARSE_ENABLE_PROJECTS
        OR "btf" IN_LIST SUITESPARSE_ENABLE_PROJECTS
        OR "camd" IN_LIST SUITESPARSE_ENABLE_PROJECTS
        OR "ccolamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    if ( NOT "suitesparse_config" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
        message ( STATUS "Adding \"suitesparse_config\" to the list of built targets." )
        list ( APPEND SUITESPARSE_ENABLE_PROJECTS "suitesparse_config" )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# include selected projects
#-------------------------------------------------------------------------------

if ( "suitesparse_config" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "SuiteSparse_config" )
    add_library ( SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig )
    if ( NOT NSTATIC )
        add_library ( SuiteSparse::SuiteSparseConfig_static ALIAS SuiteSparseConfig_static )
    endif ( )
endif ( )

if ( "mongoose" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "Mongoose" )
    add_library ( SuiteSparse::Mongoose ALIAS Mongoose )
    if ( NOT NSTATIC )
        add_library ( SuiteSparse::Mongoose_static ALIAS Mongoose_static )
    endif ( )
endif ( )

if ( "amd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "AMD" )
    add_library ( SuiteSparse::AMD ALIAS AMD )
    if ( NOT NSTATIC )
        add_library ( SuiteSparse::AMD_static ALIAS AMD_static )
    endif ( )
endif ( )

if ( "btf" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "BTF" )
    add_library ( SuiteSparse::BTF ALIAS BTF )
    if ( NOT NSTATIC )
        add_library ( SuiteSparse::BTF_static ALIAS BTF_static )
    endif ( )
endif ( )

if ( "camd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CAMD" )
    add_library ( SuiteSparse::CAMD ALIAS CAMD )
    if ( NOT NSTATIC )
        add_library ( SuiteSparse::CAMD_static ALIAS CAMD_static )
    endif ( )
endif ( )

if ( "ccolamd" IN_LIST SUITESPARSE_ENABLE_PROJECTS )
    add_subdirectory ( "CCOLAMD" )
    add_library ( SuiteSparse::CCOLAMD ALIAS CCOLAMD )
    if ( NOT NSTATIC )
        add_library ( SuiteSparse::CCOLAMD_static ALIAS CCOLAMD_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
