#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/CMakeLists.txt:  cmake for UMFPACK
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparse_config
cmake_minimum_required ( VERSION 3.22 )

set ( UMFPACK_DATE "Dec 30, 2023" )
set ( UMFPACK_VERSION_MAJOR 6 CACHE STRING "" FORCE )
set ( UMFPACK_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( UMFPACK_VERSION_SUB   0 CACHE STRING "" FORCE )

message ( STATUS "Building UMFPACK version: v"
    ${UMFPACK_VERSION_MAJOR}.
    ${UMFPACK_VERSION_MINOR}.
    ${UMFPACK_VERSION_SUB} " (" ${UMFPACK_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( umfpack
    VERSION "${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" ON )
option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." OFF )

include ( SuiteSparsePolicy )

if ( SUITESPARSE_CUDA )
    # UMFPACK with CHOLMOD (which can use CUDA)
    enable_language ( CXX CUDA )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" OFF )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

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

    if ( DEMO AND NOT WIN32 )
        find_package ( COLAMD 3.3.0
            PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::COLAMD )
            find_package ( COLAMD 3.3.0 REQUIRED )
        endif ( )

        find_package ( CAMD 3.3.0
            PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::CAMD )
            find_package ( CAMD 3.3.0 REQUIRED )
        endif ( )

        find_package ( CCOLAMD 3.3.0
            PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::CCOLAMD )
            find_package ( CCOLAMD 3.3.0 REQUIRED )
        endif ( )
    endif ( )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22

option ( NCHOLMOD "ON: do not use CHOLMOD.  OFF (default): use CHOLMOD" OFF )

if ( NOT SUITESPARSE_ROOT_CMAKELISTS AND NOT NCHOLMOD )
    # look for CHOLMOD (optional fill-reducing orderings)
    find_package ( CHOLMOD 5.1.0
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CHOLMOD )
        find_package ( CHOLMOD 5.1.0 )
    endif ( )

    if ( NOT CHOLMOD_FOUND )
        # CHOLMOD not found so disable it
        set ( NCHOLMOD true )
    endif ( )
    if ( SUITESPARSE_CUDA AND NOT CHOLMOD_CUDA_FOUND )
        set ( NCHOLMOD true )
    endif ( )
endif ( )

if ( NCHOLMOD )
    # tell UMFPACK that CHOLMOD is not available
    message ( STATUS "CHOLMOD not found or not requested" )
    add_compile_definitions ( NCHOLMOD )
else ( )
    message ( STATUS "Using CHOLMOD for addtional pre-ordering options" )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/umfpack.h.in"
    "${PROJECT_SOURCE_DIR}/Include/umfpack.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/umfpack_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/umfpack_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Source Include )

#-------------------------------------------------------------------------------
# dynamic umfpack library properties
#-------------------------------------------------------------------------------

file ( GLOB UMFPACK_SOURCES "Source2/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( UMFPACK SHARED ${UMFPACK_SOURCES} )

    set_target_properties ( UMFPACK PROPERTIES
        VERSION ${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME umfpack
        SOVERSION ${UMFPACK_VERSION_MAJOR}
        PUBLIC_HEADER "Include/umfpack.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    target_include_directories ( UMFPACK 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static umfpack library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( UMFPACK_static STATIC ${UMFPACK_SOURCES} )

    set_target_properties ( UMFPACK_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME umfpack
        PUBLIC_HEADER "Include/umfpack.h" )

    if ( MSVC )
        set_target_properties ( UMFPACK_static PROPERTIES
            OUTPUT_NAME umfpack_static )
    endif ( )

    target_include_directories ( UMFPACK_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( UMFPACK PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( UMFPACK PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( UMFPACK PRIVATE ${OpenMP_C_LIBRARIES} )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( UMFPACK_static PUBLIC ${OpenMP_C_LIBRARIES} )
        list ( APPEND UMFPACK_STATIC_LIBS ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT WIN32 )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( UMFPACK PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND UMFPACK_STATIC_LIBS "m" )
        target_link_libraries ( UMFPACK_static PUBLIC m )
    endif ( )
endif ( )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( UMFPACK PRIVATE SuiteSparse::AMD )
    target_include_directories ( UMFPACK PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::AMD,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( UMFPACK PRIVATE ${BLAS_LIBRARIES} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND UMFPACK_STATIC_LIBS ${BLAS_LIBRARIES} )
    target_link_libraries ( UMFPACK_static PUBLIC ${BLAS_LIBRARIES} )
endif ( )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# CHOLMOD:
if ( NOT NCHOLMOD )
    # link with CHOLMOD and its dependencies, both required and optional
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( UMFPACK PRIVATE SuiteSparse::CHOLMOD )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( UMFPACK_STATIC_MODULES "${UMFPACK_STATIC_MODULES} CHOLMOD" )
        if ( TARGET SuiteSparse::CHOLMOD_static )
            target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::CHOLMOD_static )
        else ( )
            target_link_libraries ( UMFPACK_static PUBLIC SuiteSparse::CHOLMOD )
        endif ( )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# UMFPACK installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS UMFPACK
        EXPORT UMFPACKTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS UMFPACK_static
        EXPORT UMFPACKTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT UMFPACKTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT UMFPACKTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/UMFPACK )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/UMFPACKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/UMFPACKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/UMFPACKConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/UMFPACK )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/UMFPACKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/UMFPACK )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( UMFPACK_STATIC_LIBS_LIST ${UMFPACK_STATIC_LIBS} )
    set ( UMFPACK_STATIC_LIBS "" )
    foreach ( _lib ${UMFPACK_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( UMFPACK_STATIC_LIBS "${UMFPACK_STATIC_LIBS} -l${_lib}" )
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
                    set ( UMFPACK_STATIC_LIBS "${UMFPACK_STATIC_LIBS} -l${_libname}" )
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
        Config/UMFPACK.pc.in
        UMFPACK.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/UMFPACK.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in UMFPACK/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( umfpack_simple  "Demo/umfpack_simple.c" )
    add_executable ( umfpack_di_demo "Demo/umfpack_di_demo.c" )
    add_executable ( umfpack_dl_demo "Demo/umfpack_dl_demo.c" )
    add_executable ( umfpack_zi_demo "Demo/umfpack_zi_demo.c" )
    add_executable ( umfpack_zl_demo "Demo/umfpack_zl_demo.c" )

    if ( NOT NFORTRAN )
        # Fortran demos
        add_executable ( umf4            "Demo/umf4.c" )
        add_executable ( umfreadhb       "Demo/readhb.f" )
        set_target_properties ( umfreadhb PROPERTIES OUTPUT_NAME readhb )
        add_executable ( umfreadhb_nozeros "Demo/readhb_nozeros.f" )
        set_target_properties ( umfreadhb_nozeros PROPERTIES OUTPUT_NAME readhb_nozeros )
        add_executable ( umfreadhb_size  "Demo/readhb_size.f" )
        set_target_properties ( umfreadhb_size PROPERTIES OUTPUT_NAME readhb_size )
        add_executable ( umf4hb          "Demo/umf4hb.f"
                                         "Demo/umf4_f77wrapper.c" )
        add_executable ( umf4zhb         "Demo/umf4zhb.f"
                                         "Demo/umf4_f77zwrapper.c" )
        add_executable ( umf4hb64        "Demo/umf4hb64.f"
                                         "Demo/umf4_f77wrapper64.c" )
        add_executable ( umf4zhb64       "Demo/umf4zhb64.f"
                                         "Demo/umf4_f77zwrapper64.c" )
    endif ( )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( umfpack_simple PUBLIC UMFPACK )
        target_link_libraries ( umfpack_di_demo PUBLIC UMFPACK )
        target_link_libraries ( umfpack_dl_demo PUBLIC UMFPACK )
        target_link_libraries ( umfpack_zi_demo PUBLIC UMFPACK )
        target_link_libraries ( umfpack_zl_demo PUBLIC UMFPACK )
    else ( )
        target_link_libraries ( umfpack_simple PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_di_demo PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_dl_demo PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_zi_demo PUBLIC UMFPACK_static )
        target_link_libraries ( umfpack_zl_demo PUBLIC UMFPACK_static )
    endif ( )
    target_link_libraries ( umfpack_simple  PUBLIC SuiteSparse::SuiteSparseConfig )
    if ( NOT NFORTRAN )
        # Fortran demos
        if ( BUILD_SHARED_LIBS )
            target_link_libraries ( umf4 PUBLIC UMFPACK )
            target_link_libraries ( umf4hb PUBLIC UMFPACK )
            target_link_libraries ( umf4zhb PUBLIC UMFPACK )
            target_link_libraries ( umf4hb64 PUBLIC UMFPACK )
            target_link_libraries ( umf4zhb64 PUBLIC UMFPACK )
        else ( )
            target_link_libraries ( umf4 PUBLIC UMFPACK_static )
            target_link_libraries ( umf4hb PUBLIC UMFPACK_static )
            target_link_libraries ( umf4zhb PUBLIC UMFPACK_static )
            target_link_libraries ( umf4hb64 PUBLIC UMFPACK_static )
            target_link_libraries ( umf4zhb64 PUBLIC UMFPACK_static )
        endif ( )
        target_link_libraries ( umf4 PUBLIC SuiteSparse::AMD )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in UMFPACK/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    include ( SuiteSparseReport )
endif ( )
