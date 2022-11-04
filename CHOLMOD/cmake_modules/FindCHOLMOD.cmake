#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindCHOLMOD.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the CHOLMOD include file and compiled library and sets:

# CHOLMOD_INCLUDE_DIR - where to find cholmod.h
# CHOLMOD_LIBRARY     - compiled CHOLMOD library
# CHOLMOD_LIBRARIES   - libraries when using CHOLMOD
# CHOLMOD_FOUND       - true if CHOLMOD found

# set ``CHOLMOD_ROOT`` to a CHOLMOD installation root to
# tell this module where to look.

# set ``SUITESPARSE_METIS_ROOT`` to a SuiteSparse_metis installation root to
# tell this module where to look for SuiteSparse_metis, a library optionally
# used by CHOLMOD.

# To use this file in your application, copy this file into MyApp/cmake_modules
# where MyApp is your application and add the following to your
# MyApp/CMakeLists.txt file:
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
#
# or, assuming MyApp and SuiteSparse sit side-by-side in a common folder, you
# can leave this file in place and use this command (revise as needed):
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules")

#-------------------------------------------------------------------------------

# include files for CHOLMOD
find_path ( CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATHS CHOLMOD_ROOT ENV CHOLMOD_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries CHOLMOD
find_library ( CHOLMOD_LIBRARY
    NAMES cholmod
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/CHOLMOD
    HINTS ${CMAKE_SOURCE_DIR}/../CHOLMOD
    PATHS CHOLMOD_ROOT ENV CHOLMOD_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
get_filename_component ( CHOLMOD_LIBRARY  ${CHOLMOD_LIBRARY} REALPATH )
get_filename_component ( CHOLMOD_FILENAME ${CHOLMOD_LIBRARY} NAME )
string (
    REGEX MATCH "[0-9]+.[0-9]+.[0-9]+"
    CHOLMOD_VERSION
    ${CHOLMOD_FILENAME}
)
set (CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} )

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( CHOLMOD
    REQUIRED_VARS CHOLMOD_LIBRARY CHOLMOD_INCLUDE_DIR
    VERSION_VAR CHOLMOD_VERSION
)

# compiled libraries used by CHOLMOD only (SuiteSparse_metis)
find_library ( SUITESPARSE_METIS_LIBRARY
    NAMES suitesparse_metis
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_metis
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse_metis
    PATHS SUITESPARSE_METIS_ROOT ENV SUITESPARSE_METIS_ROOT
    PATH_SUFFIXES lib build alternative
)

# message ( STATUS "SuiteSparse_metis: ${SUITESPARSE_METIS_LIBRARY}" )

string ( FIND ${SUITESPARSE_METIS_LIBRARY} "NOT FOUND" SMETIS )
# message ( STATUS "SFOUND: ${SMETIS}" )
if ( ${SMETIS} EQUAL -1 )
    # get the SuiteSparse_metis library
    set ( SUITESPARSE_METIS_FOUND true )
    get_filename_component ( SUITESPARSE_METIS_LIBRARY  ${SUITESPARSE_METIS_LIBRARY} REALPATH )
    # message ( STATUS "SuiteSparse_metis was found: ${SUITESPARSE_METIS_LIBRARY}" )
else ( )
    # SuiteSparse_metis not found
    set ( SUITESPARSE_METIS_FOUND false )
    # message ( STATUS "SuiteSparse_metis not found: ${SUITESPARSE_METIS_LIBRARY}" )
endif ( )

if ( SUITESPARSE_METIS_FOUND )
    message ( STATUS "SuiteSparse_metis: ${SUITESPARSE_METIS_LIBRARY}" )
    set ( CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} ${SUITESPARSE_METIS_LIBRARY} )
    # message ( STATUS "libs: ${CHOLMOD_LIBRARY}" )
endif ( )

mark_as_advanced (
    CHOLMOD_INCLUDE_DIR
    CHOLMOD_LIBRARY
    CHOLMOD_LIBRARIES
)

if ( CHOLMOD_FOUND )
    message ( STATUS "CHOLMOD include dir: ${CHOLMOD_INCLUDE_DIR}" )
    message ( STATUS "CHOLMOD library:     ${CHOLMOD_LIBRARY}" )
    message ( STATUS "CHOLMOD version:     ${CHOLMOD_VERSION}" )
    message ( STATUS "CHOLMOD libraries:   ${CHOLMOD_LIBRARIES}" )
else ( )
    message ( STATUS "CHOLMOD not found" )
endif ( )

