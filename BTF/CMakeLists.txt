#-------------------------------------------------------------------------------
# SuiteSparse/BTF/CMakeLists.txt:  cmake for BTF
#-------------------------------------------------------------------------------

# BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
# Author: Timothy A. Davis.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( BTF_DATE "Jan 17, 2023" )
set ( BTF_VERSION_MAJOR 2 )
set ( BTF_VERSION_MINOR 0 )
set ( BTF_VERSION_SUB   3 )

message ( STATUS "Building BTF version: v"
    ${BTF_VERSION_MAJOR}.
    ${BTF_VERSION_MINOR}.
    ${BTF_VERSION_SUB} " (" ${BTF_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( btf
    VERSION "${BTF_VERSION_MAJOR}.${BTF_VERSION_MINOR}.${BTF_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 7.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/btf.h.in" "${PROJECT_SOURCE_DIR}/Include/btf.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic btf library properties
#-------------------------------------------------------------------------------

file ( GLOB BTF_SOURCES "Source/*.c" )

add_library ( btf SHARED ${BTF_SOURCES} )

set_target_properties ( btf PROPERTIES
    VERSION ${BTF_VERSION_MAJOR}.${BTF_VERSION_MINOR}.${BTF_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${BTF_VERSION_MAJOR}
    PUBLIC_HEADER "Include/btf.h" 
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

#-------------------------------------------------------------------------------
# static btf library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( btf_static STATIC ${BTF_SOURCES} )

    set_target_properties ( btf_static PROPERTIES
        VERSION ${BTF_VERSION_MAJOR}.${BTF_VERSION_MINOR}.${BTF_VERSION_SUB}
        C_STANDARD_REQUIRED 11
        OUTPUT_NAME btf
        SOVERSION ${BTF_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( btf_static PROPERTIES
            OUTPUT_NAME btf_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( btf PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( btf_static PUBLIC ${SUITESPARSE_CONFIG_STATIC} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( btf PUBLIC m )
    if ( NOT NSTATIC )
        target_link_libraries ( btf_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# BTF installation location
#-------------------------------------------------------------------------------

install ( TARGETS btf
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
install ( FILES ${CMAKE_SOURCE_DIR}/cmake_modules/FindBTF.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
    COMPONENT Development )
if ( NOT NSTATIC )
    install ( TARGETS btf_static
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

