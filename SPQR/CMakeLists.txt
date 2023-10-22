#-------------------------------------------------------------------------------
# SuiteSparse/SPQR/CMakeLists.txt:  cmake for SuiteSparseQR (aka "SPQR")
#-------------------------------------------------------------------------------

# SPQR (SuiteSparseQR), Copyright (c) 2008-2023, Timothy A Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparsePolicy.cmake
cmake_minimum_required ( VERSION 3.22 )

set ( SPQR_DATE "Oct 23, 2023" )
set ( SPQR_VERSION_MAJOR 4 )
set ( SPQR_VERSION_MINOR 2 )
set ( SPQR_VERSION_SUB   2 )

message ( STATUS "Building SPQR version: v"
    ${SPQR_VERSION_MAJOR}.
    ${SPQR_VERSION_MINOR}.
    ${SPQR_VERSION_SUB} " (" ${SPQR_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" ON )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( spqr
    VERSION "${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}"
    LANGUAGES C CXX )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

option ( NOPENMP "ON (default): do not use OpenMP.  OFF: use OpenMP" ON )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

find_package ( SuiteSparse_config 7.3.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.3.0 REQUIRED )
endif ( )

find_package ( AMD 3.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::AMD )
    find_package ( AMD 3.2.1 REQUIRED )
endif ( )

find_package ( COLAMD 3.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::COLAMD )
    find_package ( COLAMD 3.2.1 REQUIRED )
endif ( )

# It would be nice if just checking for CHOLMOD would automatically pull in
# the targets for its dependencies.
find_package ( CHOLMOD 5.0.0
    PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::CHOLMOD )
    find_package ( CHOLMOD 5.0.0 REQUIRED )
endif ( )

# look for CHOLMOD's dependencies: AMD and COLAMD are required.  CAMD and
# CCOLAMD are optional, but must be found if CHOLMOD was built with them.
find_package ( CAMD 3.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::CAMD )
    find_package ( CAMD 3.2.1 )
endif ( )

find_package ( CCOLAMD 3.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::CCOLAMD )
    find_package ( CCOLAMD 3.2.1 )
endif ( )

if ( SUITESPARSE_CUDA )
    find_package ( SuiteSparse_GPURuntime 3.2.1
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_GPURuntime/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::GPURuntime )
        find_package ( SuiteSparse_GPURuntime 3.2.1 REQUIRED )
    endif ( )

    find_package ( GPUQREngine 3.2.2
        PATHS ${CMAKE_SOURCE_DIR}/../GPUQREngine/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::GPUQREngine )
        find_package ( GPUQREngine 3.2.2 REQUIRED )
    endif ( )

    find_package ( CHOLMOD_CUDA 5.0.0
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::CHOLMOD_CUDA )
        find_package ( CHOLMOD_CUDA 5.0.0 REQUIRED )
    endif ( )
endif ( )

include ( SuiteSparseBLAS )     # requires cmake 3.22
include ( SuiteSparseLAPACK )   # requires cmake 3.22

if ( SUITESPARSE_CUDA )
    # with CUDA
    add_subdirectory ( SPQRGPU )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SuiteSparseQR_definitions.h.in"
    "${PROJECT_SOURCE_DIR}/Include/SuiteSparseQR_definitions.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/spqr_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/spqr_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# dynamic spqr library properties
#-------------------------------------------------------------------------------

file ( GLOB SPQR_SOURCES "Source/spqr_*.cpp" "Source/SuiteSparseQR*.cpp" )

add_library ( SPQR SHARED ${SPQR_SOURCES} )

set_target_properties ( SPQR PROPERTIES
    VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME spqr
    SOVERSION ${SPQR_VERSION_MAJOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( SPQR 
    PRIVATE Source Include
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static spqr library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( SPQR_static STATIC ${SPQR_SOURCES} )

    set_target_properties ( SPQR_static PROPERTIES
        VERSION ${SPQR_VERSION_MAJOR}.${SPQR_VERSION_MINOR}.${SPQR_VERSION_SUB}
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME spqr
        SOVERSION ${SPQR_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( SPQR_static PROPERTIES
            OUTPUT_NAME spqr_static )
    endif ( )

    target_include_directories ( SPQR_static 
        PRIVATE Source Include
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
target_link_libraries ( SPQR PRIVATE SuiteSparse::SuiteSparseConfig )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( SPQR_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( SPQR_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    target_link_libraries ( SPQR PRIVATE OpenMP::OpenMP_CXX )
    if ( NOT NSTATIC )
        target_link_libraries ( SPQR_static PUBLIC OpenMP::OpenMP_CXX )
        list ( APPEND SPQR_STATIC_LIBS ${OpenMP_CXX_LIBRARIES} )
    endif ( )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( SPQR PRIVATE m )
    if ( NOT NSTATIC )
        list ( APPEND SPQR_STATIC_LIBS "m" )
        target_link_libraries ( SPQR_static PUBLIC m )
    endif ( )
endif ( )

# LAPACK:
message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
target_link_libraries ( SPQR PRIVATE ${LAPACK_LIBRARIES} )
target_include_directories ( SPQR PRIVATE ${LAPACK_INCLUDE_DIR} )
if ( NOT NSTATIC )
    list ( APPEND SPQR_STATIC_LIBS ${LAPACK_LIBRARIES} )
    target_link_libraries ( SPQR_static PUBLIC ${LAPACK_LIBRARIES} )
    target_include_directories ( SPQR_static PRIVATE ${LAPACK_INCLUDE_DIR} )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( SPQR PRIVATE ${BLAS_LIBRARIES} )
target_include_directories ( SPQR PRIVATE ${BLAS_INCLUDE_DIRS} )
if ( NOT NSTATIC )
    list ( APPEND SPQR_STATIC_LIBS ${BLAS_LIBRARIES} )
    target_link_libraries ( SPQR_static PUBLIC ${BLAS_LIBRARIES} )
    target_include_directories ( SPQR_static PRIVATE ${BLAS_INCLUDE_DIRS} )
endif ( )

# CHOLMOD:
# link with CHOLMOD and its dependencies
target_link_libraries ( SPQR PRIVATE
    SuiteSparse::CHOLMOD )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::CHOLMOD_static )
        target_link_libraries ( SPQR_static PUBLIC SuiteSparse::CHOLMOD_static )
    else ( )
        target_link_libraries ( SPQR_static PUBLIC SuiteSparse::CHOLMOD )
    endif ( )
endif ( )

if ( SUITESPARSE_CUDA )
    # CUDA
    target_link_libraries ( SPQR PRIVATE SuiteSparse::CHOLMOD_CUDA SPQR_CUDA )
    # SPQR includes files from the following targets that are not installed by
    # them.  That looks strange and will only work before they are installed.
    # Maybe, those two targets aren't actually standalone libraries but should
    # be part of the SPQR project?
    target_link_libraries ( SPQR PRIVATE
        SuiteSparse::GPUQREngine SuiteSparse::GPURuntime )
    target_compile_definitions ( SPQR PUBLIC "SUITESPARSE_CUDA" )
    set ( SPQR_CFLAGS "-DSUITESPARSE_CUDA" )
    if ( NOT NSTATIC )
        if ( TARGET SuiteSparse::CHOLMOD_CUDA_static )
            target_link_libraries ( SPQR_static PUBLIC SuiteSparse::CHOLMOD_CUDA_static )
        else ( )
            target_link_libraries ( SPQR_static PUBLIC SuiteSparse::CHOLMOD_CUDA )
        endif ( )
        target_link_libraries ( SPQR_static PUBLIC SPQR_CUDA_static )
        target_compile_definitions ( SPQR_static PUBLIC "SUITESPARSE_CUDA" )

        if ( TARGET SuiteSparse::GPUQREngine_static )
            target_link_libraries ( SPQR_static PUBLIC SuiteSparse::GPUQREngine_static )
        else ( )
            target_link_libraries ( SPQR_static PUBLIC SuiteSparse::GPUQREngine )
        endif ( )
        if ( TARGET SuiteSparse::GPURuntime_static )
            target_link_libraries ( SPQR_static PUBLIC SuiteSparse::GPURuntime_static )
        else ( )
            target_link_libraries ( SPQR_static PUBLIC SuiteSparse::GPURuntime )
        endif ( )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# SPQR installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS SPQR
    EXPORT SPQRTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR} )
if ( NOT NSTATIC )
    install ( TARGETS SPQR_static
        EXPORT SPQRTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

install ( FILES "Include/SuiteSparseQR_C.h"
    "Include/SuiteSparseQR_definitions.h"
    "Include/SuiteSparseQR.hpp" DESTINATION ${SUITESPARSE_INCLUDEDIR} )

# create (temporary) export target file during build
export ( EXPORT SPQRTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SPQRTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SPQRTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SPQR )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE on )
configure_package_config_file (
    Config/SPQRConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SPQRConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SPQRConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE off )
configure_package_config_file (
    Config/SPQRConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/SPQRConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SPQR )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/SPQRConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/SPQRConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SPQRConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SPQR )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( SPQR_STATIC_LIBS_LIST ${SPQR_STATIC_LIBS} )
    set ( SPQR_STATIC_LIBS "" )
    foreach ( _lib ${SPQR_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( SPQR_STATIC_LIBS "${SPQR_STATIC_LIBS} -l${_lib}" )
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
                    set ( SPQR_STATIC_LIBS "${SPQR_STATIC_LIBS} -l${_libname}" )
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
        Config/SPQR.pc.in
        SPQR.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SPQR.pc
        DESTINATION ${SUITESPARSE_LIBDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." OFF )
if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in SPQR/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( qrsimple  "Demo/qrsimple.cpp" )
    add_executable ( qrsimplec "Demo/qrsimplec.c" )
    add_executable ( qrdemo    "Demo/qrdemo.cpp" )
    add_executable ( qrdemoc   "Demo/qrdemoc.c" )

    add_executable ( qrsimplec_int32 "Demo/qrsimplec_int32.c" )
    add_executable ( qrdemo_int32    "Demo/qrdemo_int32.cpp" )
    add_executable ( qrdemoc_int32   "Demo/qrdemoc_int32.c" )


    if ( SUITESPARSE_CUDA )
        add_executable ( qrdemo_gpu  "Demo/qrdemo_gpu.cpp" )
        add_executable ( qrdemo_gpu2 "Demo/qrdemo_gpu2.cpp" )
        add_executable ( qrdemo_gpu3 "Demo/qrdemo_gpu3.cpp" )
    endif ( )

    # Libraries required for Demo programs
    target_link_libraries ( qrsimple  PUBLIC SPQR SuiteSparse::CHOLMOD )
    target_link_libraries ( qrsimplec PUBLIC SPQR SuiteSparse::CHOLMOD )
    target_link_libraries ( qrdemo    PUBLIC SPQR SuiteSparse::CHOLMOD )
    target_link_libraries ( qrdemoc   PUBLIC SPQR SuiteSparse::CHOLMOD )

    target_link_libraries ( qrsimplec_int32 PUBLIC SPQR SuiteSparse::CHOLMOD )
    target_link_libraries ( qrdemoc_int32   PUBLIC SPQR SuiteSparse::CHOLMOD )
    target_link_libraries ( qrdemo_int32    PUBLIC SPQR SuiteSparse::CHOLMOD )

    if ( SUITESPARSE_CUDA )
        target_link_libraries ( qrdemo_gpu PUBLIC SPQR SPQR_CUDA
            SuiteSparse::CHOLMOD SuiteSparse::CHOLMOD_CUDA SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( qrdemo_gpu2 PUBLIC SPQR SPQR_CUDA
            SuiteSparse::CHOLMOD SuiteSparse::CHOLMOD_CUDA SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( qrdemo_gpu3 PUBLIC SPQR SPQR_CUDA
            SuiteSparse::CHOLMOD SuiteSparse::CHOLMOD_CUDA SuiteSparse::SuiteSparseConfig )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in SPQR/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

