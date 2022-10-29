#-------------------------------------------------------------------------------
# SuiteSparse/LDL/CMakeLists.txt:  cmake for LDL
#-------------------------------------------------------------------------------

# LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( LDL_DATE "Oct 31, 2022" )  # FIXME
set ( LDL_VERSION_MAJOR 3 )
set ( LDL_VERSION_MINOR 0 )
set ( LDL_VERSION_SUB   0 )

message ( STATUS "Building LDL version: v"
    ${LDL_VERSION_MAJOR}.
    ${LDL_VERSION_MINOR}.
    ${LDL_VERSION_SUB} " (" ${LDL_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( ldl
    VERSION "${LDL_VERSION_MAJOR}.${LDL_VERSION_MINOR}.${LDL_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( AMD 3.0.0 REQUIRED )

#-------------------------------------------------------------------------------
# Configure ldl.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/ldl.h.in"
                 "${PROJECT_SOURCE_DIR}/Include/ldl.h" )
configure_file ( "Config/ldl_version.tex.in"
                 "${PROJECT_SOURCE_DIR}/Doc/ldl_version.tex" )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic ldl library properties
#-------------------------------------------------------------------------------

file ( GLOB LDL_SOURCES "Source/*.c" )

add_library ( ldl SHARED ${LDL_SOURCES} )

set_target_properties ( ldl PROPERTIES
    VERSION ${LDL_VERSION_MAJOR}.${LDL_VERSION_MINOR}.${LDL_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${LDL_VERSION_MAJOR}
    PUBLIC_HEADER "Include/ldl.h" )

#-------------------------------------------------------------------------------
# static ldl library properties
#-------------------------------------------------------------------------------

add_library ( ldl_static STATIC ${LDL_SOURCES} )

set_target_properties ( ldl_static PROPERTIES
    VERSION ${LDL_VERSION_MAJOR}.${LDL_VERSION_MINOR}.${LDL_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME ldl
    SOVERSION ${LDL_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# libm:
if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    set ( M_LIB "" )
else ( )
    set ( M_LIB "m" )
endif ( )

target_link_libraries ( ldl PUBLIC ${M_LIB}
    ${SUITESPARSE_CONFIG_LIBRARIES} ${AMD_LIBRARIES} )

target_link_libraries ( ldl_static PUBLIC ${M_LIB}
    ${SUITESPARSE_CONFIG_LIBRARIES} ${AMD_LIBRARIES} )

include_directories ( ${AMD_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# LDL installation location
#-------------------------------------------------------------------------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS ldl
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS ldl_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include," )
    message ( STATUS "  with 'make local ; make install'. No 'sudo' required." )
    install ( TARGETS ldl
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS ldl_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in LDL/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( ldlsimple      "Demo/ldlsimple.c" )
    add_executable ( ldllsimple     "Demo/ldllsimple.c" )
    add_executable ( ldlmain        "Demo/ldlmain.c" )
    add_executable ( ldllmain       "Demo/ldllmain.c" )

    # Libraries required for Demo programs
    target_link_libraries ( ldlsimple   PUBLIC ldl )
    target_link_libraries ( ldllsimple  PUBLIC ldl )
    target_link_libraries ( ldlmain     PUBLIC ldl )
    target_link_libraries ( ldllmain    PUBLIC ldl )

    if ( AMD_FOUND )
        message ( STATUS "AMD found for ldlamd and ldllamd" )
        add_executable ( ldlamd     "Demo/ldlamd.c" )
        add_executable ( ldllamd    "Demo/ldllamd.c" )
        target_link_libraries ( ldlamd  PUBLIC ldl )
        target_link_libraries ( ldllamd PUBLIC ldl )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in LDL/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

