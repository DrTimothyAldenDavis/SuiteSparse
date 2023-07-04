#-------------------------------------------------------------------------------
# SuiteSparse/UMFPACK/CMakeLists.txt:  cmake for UMFPACK
#-------------------------------------------------------------------------------

# Copyright (c) 1995-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS in SuiteSparse_config
cmake_minimum_required ( VERSION 3.22 )

set ( UMFPACK_DATE "June 16, 2023" )
set ( UMFPACK_VERSION_MAJOR 6 )
set ( UMFPACK_VERSION_MINOR 1 )
set ( UMFPACK_VERSION_SUB   1 )

message ( STATUS "Building UMFPACK version: v"
    ${UMFPACK_VERSION_MAJOR}.
    ${UMFPACK_VERSION_MINOR}.
    ${UMFPACK_VERSION_SUB} " (" ${UMFPACK_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../CHOLMOD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( umfpack
    VERSION "${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" off )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

find_package ( SuiteSparse_config 7.1.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::suitesparseconfig )
    find_package ( SuiteSparse_config 7.1.0 REQUIRED )
endif ( )
find_package ( AMD 3.0.4
    PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
if ( NOT TARGET amd )
    find_package ( AMD 3.0.4 REQUIRED )
endif ( )
include ( SuiteSparseBLAS )     # requires cmake 3.22

option ( NCHOLMOD "ON: do not use CHOLMOD.  OFF (default): use CHOLMOD" off )

if ( NOT NCHOLMOD )
    # look for CHOLMOD (optional fill-reducing orderings)
    find_package ( CHOLMOD 4.0.4
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET cholmod )
        find_package ( CHOLMOD 4.0.4 )
    endif ( )

    find_package ( CHOLMOD_CUDA 4.0.4
        PATHS ${CMAKE_SOURCE_DIR}/../CHOLMOD/build NO_DEFAULT_PATH )
    if ( NOT TARGET cholmod_cuda )
        find_package ( CHOLMOD_CUDA 4.0.4 REQUIRED )
    endif ( )

    # look for CHOLMOD's dependencies: AMD and COLAMD are required.  CAMD and
    # CCOLAMD are optional, but must be found if CHOLMOD was built with them.
    find_package ( COLAMD 3.0.4
        PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET colamd )
        find_package ( COLAMD 3.0.4 )
    endif ( )

    find_package ( CAMD 3.0.4
        PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET camd )
        find_package ( CAMD 3.0.4 )
    endif ( )

    find_package ( CCOLAMD 3.0.4
        PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET ccolamd )
        find_package ( CCOLAMD 3.0.4 )
    endif ( )

    if ( NOT CHOLMOD_FOUND OR NOT AMD_FOUND OR NOT COLAMD_FOUND )
        # CHOLMOD not found so disable it
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

add_library ( umfpack SHARED ${UMFPACK_SOURCES} )

set_target_properties ( umfpack PROPERTIES
    VERSION ${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    SOVERSION ${UMFPACK_VERSION_MAJOR}
    PUBLIC_HEADER "Include/umfpack.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

target_include_directories ( umfpack 
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

#-------------------------------------------------------------------------------
# static umfpack library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( umfpack_static STATIC ${UMFPACK_SOURCES} )

    set_target_properties ( umfpack_static PROPERTIES
        VERSION ${UMFPACK_VERSION_MAJOR}.${UMFPACK_VERSION_MINOR}.${UMFPACK_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME umfpack
        SOVERSION ${UMFPACK_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( umfpack_static PROPERTIES
            OUTPUT_NAME umfpack_static )
    endif ( )

    target_include_directories ( umfpack_static 
        INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( umfpack PRIVATE SuiteSparse::suitesparseconfig )
target_include_directories ( umfpack PUBLIC "$<TARGET_PROPERTY:SuiteSparse::suitesparseconfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::suitesparseconfig_static )
        target_link_libraries ( umfpack_static PUBLIC SuiteSparse::suitesparseconfig_static )
    else ( )
        target_link_libraries ( umfpack_static PUBLIC SuiteSparse::suitesparseconfig )
    endif ( )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( umfpack PRIVATE ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
        target_link_libraries ( umfpack_static PUBLIC ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( umfpack PRIVATE m )
    if ( NOT NSTATIC )
        target_link_libraries ( umfpack_static PUBLIC m )
    endif ( )
endif ( )

# amd:
target_link_libraries ( umfpack PRIVATE amd )
target_include_directories ( umfpack PUBLIC "$<TARGET_PROPERTY:amd,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET amd_static )
        target_link_libraries ( umfpack_static PUBLIC amd_static )
    else ( )
        target_link_libraries ( umfpack_static PUBLIC amd )
    endif ( )
endif ( )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( umfpack PRIVATE ${BLAS_LIBRARIES} )
if ( NOT NSTATIC )
    target_link_libraries ( umfpack_static PUBLIC ${BLAS_LIBRARIES} )
endif ( )
include_directories ( ${BLAS_INCLUDE_DIRS} )

# cholmod:
if ( NOT NCHOLMOD )
    # link with CHOLMOD and its dependencies, both required and optional
    target_link_libraries ( umfpack PRIVATE
        cholmod ${CHOLMOD_CUDA_LIBRARIES} )
    if ( NOT NSTATIC )
        target_link_libraries ( umfpack_static PUBLIC
            ${CHOLMOD_CUDA_STATIC} )
        if ( TARGET cholmod_static )
            target_link_libraries ( umfpack_static PUBLIC cholmod_static )
        else ( )
            target_link_libraries ( umfpack_static PUBLIC cholmod )
        endif ( )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# UMFPACK installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS umfpack
    EXPORT UMFPACKTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS umfpack_static
        EXPORT UMFPACKTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT UMFPACKTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT UMFPACKTargets
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/UMFPACK )

configure_package_config_file (
    Config/UMFPACKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/UMFPACK )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/UMFPACKConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/UMFPACK )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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
        add_executable ( readhb          "Demo/readhb.f" )
        add_executable ( readhb_nozeros  "Demo/readhb_nozeros.f" )
        add_executable ( readhb_size     "Demo/readhb_size.f" )
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
    target_link_libraries ( umfpack_simple  PUBLIC umfpack SuiteSparse::suitesparseconfig )
    target_link_libraries ( umfpack_di_demo PUBLIC umfpack )
    target_link_libraries ( umfpack_dl_demo PUBLIC umfpack )
    target_link_libraries ( umfpack_zi_demo PUBLIC umfpack )
    target_link_libraries ( umfpack_zl_demo PUBLIC umfpack )
    if ( NOT WIN32 )
        # This is only needed because the demos might be linked before those
        # libraries are installed. Would it be better to set the rpath instead?
        target_link_libraries ( umfpack_di_demo PUBLIC
            colamd camd ccolamd )
        target_link_libraries ( umfpack_dl_demo PUBLIC
            colamd camd ccolamd )
        target_link_libraries ( umfpack_zi_demo PUBLIC
            colamd camd ccolamd )
        target_link_libraries ( umfpack_zl_demo PUBLIC
            colamd camd ccolamd )
    endif ( )
    if ( NOT NFORTRAN )
        # Fortran demos
        target_link_libraries ( umf4            PUBLIC umfpack amd )
        target_link_libraries ( umf4hb          PUBLIC umfpack )
        target_link_libraries ( umf4zhb         PUBLIC umfpack )
        target_link_libraries ( umf4hb64        PUBLIC umfpack )
        target_link_libraries ( umf4zhb64       PUBLIC umfpack )
        if ( NOT WIN32 )
            # This is only needed because the demos might be linked before
            # those libraries are installed. Would it be better to set the
            # rpath instead?
            target_link_libraries ( umf4hb PUBLIC
                colamd camd ccolamd )
            target_link_libraries ( umf4zhb PUBLIC
                colamd camd ccolamd )
            target_link_libraries ( umf4hb64 PUBLIC
                colamd camd ccolamd )
            target_link_libraries ( umf4zhb64 PUBLIC
                colamd camd ccolamd )
        endif ( )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in UMFPACK/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

