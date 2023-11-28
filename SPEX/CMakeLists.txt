#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/CMakeLists.txt:  cmake for SPEX
#-------------------------------------------------------------------------------

# Copyright (c) 1996-2023, Timothy A. Davis, Patrick Amestoy, Iain Duff.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( SPEX_DATE "Dec 30, 2023" )
set ( SPEX_VERSION_MAJOR 2 CACHE STRING "" FORCE )
set ( SPEX_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( SPEX_VERSION_SUB   0 CACHE STRING "" FORCE )

message ( STATUS "Building SPEX version: v"
    ${SPEX_VERSION_MAJOR}.
    ${SPEX_VERSION_MINOR}.
    ${SPEX_VERSION_SUB} " (" ${SPEX_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( spex
    VERSION "${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.4.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.4.0 REQUIRED )
    endif ( )

    find_package ( AMD 3.3.0
        PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::AMD )
        find_package ( AMD 3.3.0 REQUIRED )
    endif ( )

    find_package ( COLAMD 3.3.0
        PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::COLAMD )
        find_package ( COLAMD 3.3.0 REQUIRED )
    endif ( )
endif ( )

find_package ( GMP 6.1.2 REQUIRED )     # from SPEX/cmake_modules
find_package ( MPFR 4.0.2 REQUIRED )    # from SPEX/cmake_modules

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SPEX.h.in"
    "${PROJECT_SOURCE_DIR}/Include/SPEX.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/SPEX_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/SPEX_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( SPEX_Left_LU/Source SPEX_Util/Source Include 
    SPEX_Left_LU/Demo
    ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic spex library properties
#-------------------------------------------------------------------------------

file ( GLOB SPEX_SOURCES "SPEX*/Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( SPEX SHARED ${SPEX_SOURCES} )

    set_target_properties ( SPEX PROPERTIES
        VERSION ${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME spex
        SOVERSION ${SPEX_VERSION_MAJOR}
        PUBLIC_HEADER "Include/SPEX.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    target_include_directories ( SPEX 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static spex library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( SPEX_static STATIC ${SPEX_SOURCES} )

    set_target_properties ( SPEX_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME spex
        PUBLIC_HEADER "Include/SPEX.h" )

    if ( MSVC )
        set_target_properties ( SPEX_static PROPERTIES
            OUTPUT_NAME spex_static )
    endif ( )

    target_include_directories ( SPEX_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( SPEX PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE SuiteSparse::AMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# COLAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE SuiteSparse::COLAMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::COLAMD_static )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::COLAMD_static )
    else ( )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::COLAMD )
    endif ( )
endif ( )

# MPFR:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE ${MPFR_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND SPEX_STATIC_LIBS ${MPFR_STATIC} )
    target_link_libraries ( SPEX_static PUBLIC ${MPFR_STATIC} )
endif ( )

# GMP:
# must occur after MPFR
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE ${GMP_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND SPEX_STATIC_LIBS ${GMP_STATIC} )
    target_link_libraries ( SPEX_static PUBLIC ${GMP_STATIC} )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( SPEX PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND SPEX_STATIC_LIBS "m" )
        target_link_libraries ( SPEX_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# SPEX installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS SPEX
        EXPORT SPEXTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS SPEX_static
        EXPORT SPEXTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT SPEXTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SPEXTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SPEXTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SPEX )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/SPEXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/SPEXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/SPEXConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SPEX )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/SPEXConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindGMP.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindMPFR.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SPEX )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( SPEX_STATIC_LIBS_LIST ${SPEX_STATIC_LIBS} )
    set ( SPEX_STATIC_LIBS "" )
    foreach ( _lib ${SPEX_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( SPEX_STATIC_LIBS "${SPEX_STATIC_LIBS} -l${_lib}" )
            continue ()
        endif ( )
        set ( _kinds "SHARED" "STATIC" )
        if ( WIN32 )
            list ( PREPEND _kinds "IMPORT" )
        endif ( )
        foreach ( _kind IN LISTS _kinds )
            set ( _regex ".*\\/(lib)?([^\\.]*)(${CMAKE_${_kind}_LIBRARY_SUFFIX})" )
            if ( ${_lib} MATCHES ${_regex} )
                string ( REGEX REPLACE ${_regex} "\\2" _libname ${_lib} )
                if ( NOT "${_libname}" STREQUAL "" )
                    set ( SPEX_STATIC_LIBS "${SPEX_STATIC_LIBS} -l${_libname}" )
                    break ()
                endif ( )
            endif ( )
        endforeach ( )
    endforeach ( )

    set ( prefix "${CMAKE_INSTALL_PREFIX}" )
    set ( exec_prefix "\${prefix}" )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
    if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
        set ( libdir "${SUITESPARSE_LIBDIR}")
    else ( )
        set ( libdir "\${exec_prefix}/${SUITESPARSE_LIBDIR}")
    endif ( )
    cmake_path ( IS_ABSOLUTE SUITESPARSE_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
    if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
        set ( includedir "${SUITESPARSE_INCLUDEDIR}")
    else ( )
        set ( includedir "\${prefix}/${SUITESPARSE_INCLUDEDIR}")
    endif ( )
    configure_file (
        Config/SPEX.pc.in
        SPEX.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SPEX.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in SPEX/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( spexlu_demo "SPEX_Left_LU/Demo/spexlu_demo.c"
                                 "SPEX_Left_LU/Demo/demos.c" )
    add_executable ( example     "SPEX_Left_LU/Demo/example.c" )
    add_executable ( example2    "SPEX_Left_LU/Demo/example2.c"
                                 "SPEX_Left_LU/Demo/demos.c" )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( spexlu_demo PUBLIC SPEX )
        target_link_libraries ( example PUBLIC SPEX )
        target_link_libraries ( example2 PUBLIC SPEX )
    else ( )
        target_link_libraries ( spexlu_demo PUBLIC SPEX_static )
        target_link_libraries ( example PUBLIC SPEX_static )
        target_link_libraries ( example2 PUBLIC SPEX_static )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in SPEX/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    include ( SuiteSparseReport )
endif ( )
