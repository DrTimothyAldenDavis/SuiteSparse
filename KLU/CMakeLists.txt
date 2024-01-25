#-------------------------------------------------------------------------------
# SuiteSparse/KLU/CMakeLists.txt:  cmake for KLU
#-------------------------------------------------------------------------------

# KLU, Copyright (c) 2004-2023, University of Florida.  All Rights Reserved.
# Authors: Timothy A. Davis and Ekanathan Palamadai.
# SPDX-License-Identifier: LGPL-2.1+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

set ( KLU_DATE "Jan 20, 2024" )
set ( KLU_VERSION_MAJOR 2 CACHE STRING "" FORCE )
set ( KLU_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( KLU_VERSION_SUB   2 CACHE STRING "" FORCE )

message ( STATUS "Building KLU version: v"
    ${KLU_VERSION_MAJOR}.
    ${KLU_VERSION_MINOR}.
    ${KLU_VERSION_SUB} " (" ${KLU_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( KLU
    VERSION "${KLU_VERSION_MAJOR}.${KLU_VERSION_MINOR}.${KLU_VERSION_SUB}"
    LANGUAGES C )
#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.6.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.6.0 REQUIRED )
    endif ( )

    find_package ( AMD 3.3.1
        PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::AMD )
        find_package ( AMD 3.3.1 REQUIRED )
    endif ( )

    find_package ( COLAMD 3.3.2
        PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::COLAMD )
        find_package ( COLAMD 3.3.2 REQUIRED )
    endif ( )

    find_package ( BTF 2.3.1
        PATHS ${CMAKE_SOURCE_DIR}/../BTF/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::BTF )
        find_package ( BTF 2.3.1 REQUIRED )
    endif ( )
endif ( )

option ( KLU_USE_CHOLMOD "ON (default): use CHOLMOD in KLU.  OFF: do not use CHOLMOD in KLU" ON )

if ( SUITESPARSE_ROOT_CMAKELISTS )
    # if KLU_USE_CHOLMOD is true, then CHOLMOD has been added to the
    # list of packages to compile in the root CMakeLists.txt.
    set ( KLU_HAS_CHOLMOD ${KLU_USE_CHOLMOD} )
else ( )
    if ( KLU_USE_CHOLMOD )
        # look for CHOLMOD (optional fill-reducing orderings)
        find_package ( CHOLMOD 5.2.0
            PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::CHOLMOD )
            find_package ( CHOLMOD 5.2.0 )
        endif ( )
        if ( NOT CHOLMOD_FOUND )
            # CHOLMOD not found so disable it
            set ( KLU_HAS_CHOLMOD OFF )
        else ( )
            set ( KLU_HAS_CHOLMOD ON )
        endif ( )
    else ( )
        set ( KLU_HAS_CHOLMOD OFF )
    endif ( )
endif ( )

if ( KLU_HAS_CHOLMOD )
    message ( STATUS "Using CHOLMOD for addtional pre-ordering options" )
else ( )
    add_compile_definitions ( NCHOLMOD )
    message ( STATUS "CHOLMOD not found or not requested" )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND KLU_USE_CHOLMOD AND NOT KLU_HAS_CHOLMOD )
    message ( FATAL_ERROR "CHOLMOD required for KLU but not found" )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/klu.h.in"
    "${PROJECT_SOURCE_DIR}/Include/klu.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/klu_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/klu_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include User )

#-------------------------------------------------------------------------------
# dynamic klu library properties
#-------------------------------------------------------------------------------

file ( GLOB KLU_SOURCES "Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( KLU SHARED ${KLU_SOURCES} )

    set_target_properties ( KLU PROPERTIES
        VERSION ${KLU_VERSION_MAJOR}.${KLU_VERSION_MINOR}.${KLU_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME klu
        SOVERSION ${KLU_VERSION_MAJOR}
        PUBLIC_HEADER "Include/klu.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( KLU PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( KLU
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static klu library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( KLU_static STATIC ${KLU_SOURCES} )

    set_target_properties ( KLU_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME klu
        PUBLIC_HEADER "Include/klu.h" )

    if ( MSVC )
        set_target_properties ( KLU_static PROPERTIES
            OUTPUT_NAME klu_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( KLU_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( KLU_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# klu_cholmod library properties
#-------------------------------------------------------------------------------

if ( KLU_HAS_CHOLMOD )

    file ( GLOB KLU_CHOLMOD_SOURCES "User/*.c" )

    if ( BUILD_SHARED_LIBS )
        add_library ( KLU_CHOLMOD SHARED ${KLU_CHOLMOD_SOURCES} )

        set_target_properties ( KLU_CHOLMOD PROPERTIES
            VERSION ${KLU_VERSION_MAJOR}.${KLU_VERSION_MINOR}.${KLU_VERSION_SUB}
            C_STANDARD 11
            C_STANDARD_REQUIRED ON
            OUTPUT_NAME klu_cholmod
            SOVERSION ${KLU_VERSION_MAJOR}
            PUBLIC_HEADER "User/klu_cholmod.h" )

        if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
            set_target_properties ( KLU_CHOLMOD PROPERTIES EXPORT_NO_SYSTEM ON )
        endif ( )

        target_include_directories ( KLU_CHOLMOD
            INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                      $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
    endif ( )

    if ( BUILD_STATIC_LIBS )
        add_library ( KLU_CHOLMOD_static STATIC ${KLU_CHOLMOD_SOURCES} )

        set_target_properties ( KLU_CHOLMOD_static PROPERTIES
            C_STANDARD 11
            C_STANDARD_REQUIRED ON
            OUTPUT_NAME klu_cholmod
            PUBLIC_HEADER "User/klu_cholmod.h" )

        if ( MSVC )
            set_target_properties ( KLU_CHOLMOD_static PROPERTIES
                OUTPUT_NAME klu_cholmod_static )
        endif ( )

        if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
            set_target_properties ( KLU_CHOLMOD_static PROPERTIES EXPORT_NO_SYSTEM ON )
        endif ( )

        target_include_directories ( KLU_CHOLMOD_static
            INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                      $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
    endif ( )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( KLU PRIVATE SuiteSparse::SuiteSparseConfig )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( KLU_static PRIVATE SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( KLU_static PRIVATE SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( KLU PRIVATE SuiteSparse::AMD )
    target_include_directories ( KLU PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::AMD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( KLU_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( KLU_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# COLAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( KLU PRIVATE SuiteSparse::COLAMD )
    target_include_directories ( KLU PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::COLAMD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::COLAMD_static )
        target_link_libraries ( KLU_static PUBLIC SuiteSparse::COLAMD_static )
    else ( )
        target_link_libraries ( KLU_static PUBLIC SuiteSparse::COLAMD )
    endif ( )
endif ( )

# BTF:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( KLU PRIVATE SuiteSparse::BTF )
    target_include_directories ( KLU PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::BTF,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::BTF_static )
        target_link_libraries ( KLU_static PUBLIC SuiteSparse::BTF_static )
    else ( )
        target_link_libraries ( KLU_static PUBLIC SuiteSparse::BTF )
    endif ( )
endif ( )

if ( KLU_HAS_CHOLMOD )

    # CHOLMOD:
    # link with CHOLMOD and its dependencies, both required and optional
    # CHOLMOD without CUDA
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( KLU_CHOLMOD PRIVATE SuiteSparse::CHOLMOD )
    endif ( )
    if ( BUILD_STATIC_LIBS )
       set ( KLU_STATIC_MODULES "${KLU_STATIC_MODULES} CHOLMOD" )
       if ( TARGET SuiteSparse::CHOLMOD_static )
            target_link_libraries ( KLU_CHOLMOD_static PRIVATE SuiteSparse::CHOLMOD_static )
        else ( )
            target_link_libraries ( KLU_CHOLMOD_static PRIVATE SuiteSparse::CHOLMOD )
        endif ( )
    endif ( )

    # klu:
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( KLU_CHOLMOD PRIVATE KLU )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( KLU_CHOLMOD_static PRIVATE KLU_static )
        if ( TARGET SuiteSparse::BTF_static )
            target_link_libraries ( KLU_CHOLMOD_static PRIVATE SuiteSparse::BTF_static )
        else ( )
            target_link_libraries ( KLU_CHOLMOD_static PRIVATE SuiteSparse::BTF )
        endif ( )
    endif ( )

endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( KLU PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( KLU_STATIC_LIBS "${KLU_STATIC_LIBS} -lm" )
        target_link_libraries ( KLU_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# KLU installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS KLU
        EXPORT KLUTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

if ( BUILD_STATIC_LIBS )
    install ( TARGETS KLU_static
        EXPORT KLUTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT KLUTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/KLUTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT KLUTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/KLU )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/KLUConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/KLUConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/KLUConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/KLUConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/KLUConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/KLU )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/KLUConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/KLUConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/KLUConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/KLU )

#-------------------------------------------------------------------------------
# create pkg-config file for KLU
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
    if ( BUILD_SHARED_LIBS )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:KLU> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:KLU_static> )
    endif ( )
    configure_file (
        Config/KLU.pc.in
        KLU.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT KLU.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/KLU.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/KLU.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# KLU_CHOLMOD installation
#-------------------------------------------------------------------------------

if ( KLU_HAS_CHOLMOD )
    if ( BUILD_SHARED_LIBS )
        install ( TARGETS KLU_CHOLMOD
            EXPORT KLU_CHOLMODTargets
            LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
            ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
            RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
            PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        install ( TARGETS KLU_CHOLMOD_static
            EXPORT KLU_CHOLMODTargets
            ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
            PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    endif ( )

    # create (temporary) export target file during build
    export ( EXPORT KLU_CHOLMODTargets
        NAMESPACE SuiteSparse::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMODTargets.cmake )

    # install export target, config and version files for find_package
    install ( EXPORT KLU_CHOLMODTargets
        NAMESPACE SuiteSparse::
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/KLU_CHOLMOD )

    # generate config file to be used in common build tree
    set ( SUITESPARSE_IN_BUILD_TREE ON )
    configure_package_config_file (
        Config/KLU_CHOLMODConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMODConfig.cmake
        INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMODConfig.cmake )

    # generate config file to be installed
    set ( SUITESPARSE_IN_BUILD_TREE OFF )
    configure_package_config_file (
        Config/KLU_CHOLMODConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/target/KLU_CHOLMODConfig.cmake
        INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/KLU_CHOLMOD )

    write_basic_package_version_file (
        ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMODConfigVersion.cmake
        COMPATIBILITY SameMajorVersion )

    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMODConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMODConfigVersion.cmake
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/KLU_CHOLMOD )

    #---------------------------------------------------------------------------
    # create pkg-config file for KLU_CHOLMOD
    #---------------------------------------------------------------------------

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
        if ( BUILD_SHARED_LIBS )
            set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:KLU_CHOLMOD> )
        else ( )
            set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:KLU_CHOLMOD_static> )
        endif ( )
        configure_file (
            Config/KLU_CHOLMOD.pc.in
            KLU_CHOLMOD.pc.out
            @ONLY
            NEWLINE_STYLE LF )
        file ( GENERATE
            OUTPUT KLU_CHOLMOD.pc
            INPUT ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMOD.pc.out
            NEWLINE_STYLE LF )
        install ( FILES
            ${CMAKE_CURRENT_BINARY_DIR}/KLU_CHOLMOD.pc
            DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in KLU/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( klu_simple  "Demo/klu_simple.c" )
    if ( KLU_HAS_CHOLMOD )
        add_executable ( kludemo     "Demo/kludemo.c" )
        add_executable ( kluldemo    "Demo/kluldemo.c" )
    endif ( )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( klu_simple PUBLIC KLU )
    else ( )
        target_link_libraries ( klu_simple PUBLIC KLU_static )
    endif ( )
    if ( KLU_HAS_CHOLMOD )
        if ( BUILD_SHARED_LIBS )
            target_link_libraries ( kludemo PUBLIC KLU_CHOLMOD KLU SuiteSparse::CHOLMOD )
            target_link_libraries ( kluldemo PUBLIC KLU_CHOLMOD KLU SuiteSparse::CHOLMOD )
        else ( )
            target_link_libraries ( kludemo PUBLIC KLU_CHOLMOD_static KLU_static SuiteSparse::CHOLMOD )
            target_link_libraries ( kluldemo PUBLIC KLU_CHOLMOD_static KLU_static SuiteSparse::CHOLMOD )
        endif ( )
        if ( NOT WIN32 )
            target_link_libraries ( kludemo PUBLIC m )
            target_link_libraries ( kluldemo PUBLIC m )
        endif ( )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in KLU/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
