#-------------------------------------------------------------------------------
# SuiteSparse/LDL/CMakeLists.txt:  cmake for LDL
#-------------------------------------------------------------------------------

# LDL, Copyright (c) 2005-2022 by Timothy A. Davis. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( LDL_DATE "Sept 18, 2023" )
set ( LDL_VERSION_MAJOR 3 )
set ( LDL_VERSION_MINOR 2 )
set ( LDL_VERSION_SUB   1 )

message ( STATUS "Building LDL version: v"
    ${LDL_VERSION_MAJOR}.
    ${LDL_VERSION_MINOR}.
    ${LDL_VERSION_SUB} " (" ${LDL_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

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

find_package ( SuiteSparse_config 7.2.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.2.0 REQUIRED )
endif ( )
find_package ( AMD 3.2.0
    PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::AMD )
    find_package ( AMD 3.2.0 REQUIRED )
endif ( )

#-------------------------------------------------------------------------------
# Configure ldl.h with version number
#-------------------------------------------------------------------------------

configure_file ( "Config/ldl.h.in"
    "${PROJECT_SOURCE_DIR}/Include/ldl.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/ldl_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/ldl_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic ldl library properties
#-------------------------------------------------------------------------------

file ( GLOB LDL_SOURCES "Source/*.c" )

add_library ( LDL SHARED ${LDL_SOURCES} )

set_target_properties ( LDL PROPERTIES
    VERSION ${LDL_VERSION_MAJOR}.${LDL_VERSION_MINOR}.${LDL_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME ldl
    SOVERSION ${LDL_VERSION_MAJOR}
    PUBLIC_HEADER "Include/ldl.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( LDL 
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static ldl library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( LDL_static STATIC ${LDL_SOURCES} )

    set_target_properties ( LDL_static PROPERTIES
        VERSION ${LDL_VERSION_MAJOR}.${LDL_VERSION_MINOR}.${LDL_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME ldl
        SOVERSION ${LDL_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( LDL_static PROPERTIES
            OUTPUT_NAME ldl_static )
    endif ( )

    target_include_directories ( LDL_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# libm:
if ( NOT WIN32 )
    target_link_libraries ( LDL PRIVATE m )
    if ( NOT NSTATIC )
        set ( LDL_STATIC_LIBS "${LDL_STATIC_LIBS} -lm" )
        target_link_libraries ( LDL_static PUBLIC m )
    endif ( )
endif ( )

target_link_libraries ( LDL PRIVATE
    SuiteSparse::AMD SuiteSparse::SuiteSparseConfig )
target_include_directories ( LDL PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )

if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( LDL_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( LDL_static PUBLIC SuiteSparse::AMD )
    endif ( )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( LDL_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( LDL_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )


#-------------------------------------------------------------------------------
# LDL installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS LDL
    EXPORT LDLTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS LDL_static
        EXPORT LDLTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT LDLTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/LDLTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT LDLTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/LDL )

configure_package_config_file (
    Config/LDLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LDLConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/LDL )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/LDLConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LDLConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LDLConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/LDL )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
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
        Config/LDL.pc.in
        LDL.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/LDL.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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
    target_link_libraries ( ldlsimple   PUBLIC LDL SuiteSparse::SuiteSparseConfig )
    target_link_libraries ( ldllsimple  PUBLIC LDL )
    target_link_libraries ( ldlmain     PUBLIC LDL )
    target_link_libraries ( ldllmain    PUBLIC LDL )

    if ( AMD_FOUND )
        message ( STATUS "AMD found for ldlamd and ldllamd" )
        add_executable ( ldlamd     "Demo/ldlamd.c" )
        add_executable ( ldllamd    "Demo/ldllamd.c" )
        target_link_libraries ( ldlamd  PUBLIC LDL SuiteSparse::AMD )
        target_link_libraries ( ldllamd PUBLIC LDL SuiteSparse::AMD )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in LDL/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

