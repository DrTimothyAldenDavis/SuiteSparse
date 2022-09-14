#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/CMakeLists.txt:  cmake for CHOLMOD
#-------------------------------------------------------------------------------

# CHOLMOD: Copyright (c) 2005-2022, Timothy A. Davis.
# Copyright and license varies by module.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.19 )

set ( CHOLMOD_DATE "Sept FIXME, 2022" )
set ( CHOLMOD_VERSION_MAJOR 4 )
set ( CHOLMOD_VERSION_MINOR 0 )
set ( CHOLMOD_VERSION_SUB   0 )

message ( STATUS "Building CHOLMOD version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/cmake_modules
    )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( cholmod
    VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
    LANGUAGES C ) # <-- add CXX here for packages that use C++

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C Fortran )
find_package ( OpenMP )
find_package ( SuiteSparse_config REQUIRED )
find_package ( COLAMD REQUIRED )
find_package ( AMD REQUIRED )
find_package ( BLAS REQUIRED )
find_package ( LAPACK REQUIRED )
find_package ( CAMD )
find_package ( CCOLAMD )
find_package ( METIS )

# no gpu yet
# add_compile_definitions ( NGPU )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/cholmod.h.in" "${PROJECT_SOURCE_DIR}/Include/cholmod.h")
configure_file ( "Config/cholmod_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/cholmod_version.tex")

#-------------------------------------------------------------------------------
# report status and set compile flags
#-------------------------------------------------------------------------------

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "CMAKE C Flags debug:     ${CMAKE_C_FLAGS_DEBUG} ")
#   message ( STATUS "CMAKE C++ Flags debug:   ${CMAKE_CXX_FLAGS_DEBUG} ")
else ( )
    message ( STATUS "CMAKE C Flags release:   ${CMAKE_C_FLAGS_RELEASE} ")
#   message ( STATUS "CMAKE C++ Flags release: ${CMAKE_CXX_FLAGS_RELEASE} ")
endif ( )

message ( STATUS "CMAKE C compiler:        ${CMAKE_C_COMPILER_ID} ")
message ( STATUS "CMAKE have OpenMP:       ${OPENMP_FOUND} ")

# message ( STATUS "CMAKE C++ compiler:      " ${CMAKE_CXX_COMPILER_ID} )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Check Cholesky Core MatrixOps Modify Partition
    Supernodal Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} ${AMD_INCLUDE_DIR}
    ${COLAMD_INCLUDE_DIR} ${CAMD_INCLUDE_DIR} ${CCOLAMD_INCLUDE_DIR}
    ${METIS_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic cholmod library properties
#-------------------------------------------------------------------------------

file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
    "Core/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
    "Partition/cholmod_*.c" "Supernodal/cholmod_*.c")

add_library ( cholmod SHARED ${CHOLMOD_SOURCES} )

SET_TARGET_PROPERTIES ( cholmod PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CHOLMOD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cholmod.h" )

#-------------------------------------------------------------------------------
# static cholmod library properties
#-------------------------------------------------------------------------------

add_library ( cholmod_static STATIC ${CHOLMOD_SOURCES} )

SET_TARGET_PROPERTIES ( cholmod_static PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME cholmod
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( cholmod PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
target_link_libraries ( cholmod_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( cholmod PUBLIC ${OpenMP_C_LIBRARIES} )
    target_link_libraries ( cholmod_static PUBLIC ${OpenMP_C_LIBRARIES} )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( cholmod PUBLIC m )
    target_link_libraries ( cholmod_static PUBLIC m )
endif ( )

# amd:
target_link_libraries ( cholmod PUBLIC ${AMD_LIBRARIES} )
target_link_libraries ( cholmod_static PUBLIC ${AMD_LIBRARIES} )

# colamd:
target_link_libraries ( cholmod PUBLIC ${COLAMD_LIBRARIES} )
target_link_libraries ( cholmod_static PUBLIC ${COLAMD_LIBRARIES} )

# BLAS:
message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
target_link_libraries ( cholmod PUBLIC ${BLAS_LIBRARIES} )
target_link_libraries ( cholmod_static PUBLIC ${BLAS_LIBRARIES} )

# LAPACK:
message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
target_link_libraries ( cholmod PUBLIC ${LAPACK_LIBRARIES} )
target_link_libraries ( cholmod_static PUBLIC ${LAPACK_LIBRARIES} )

# camd:
if ( CAMD_FOUND )
    target_link_libraries ( cholmod PUBLIC ${CAMD_LIBRARIES} )
    target_link_libraries ( cholmod_static PUBLIC ${CAMD_LIBRARIES} )
endif ( )

# ccolamd:
if ( CCOLAMD_FOUND )
    target_link_libraries ( cholmod PUBLIC ${CCOLAMD_LIBRARIES} )
    target_link_libraries ( cholmod_static PUBLIC ${CCOLAMD_LIBRARIES} )
endif ( )

# metis:
if ( METIS_FOUND )
    target_link_libraries ( cholmod PUBLIC ${METIS_LIBRARIES} )
    target_link_libraries ( cholmod_static PUBLIC ${METIS_LIBRARIES} )
endif ( )

#-------------------------------------------------------------------------------
# CHOLMOD installation location
#---------------------------------------------------------------Z--------------

# use "cmake -DLOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT LOCAL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS cholmod
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    install ( TARGETS cholmod_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS cholmod
        LIBRARY       DESTINATION ${LOCAL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${LOCAL_INCLUDEDIR} )
    install ( TARGETS cholmod_static
        ARCHIVE       DESTINATION ${LOCAL_LIBDIR} )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( DEMO )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in CHOLMOD/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    add_executable ( cholmod_demo   "Demo/cholmod_demo.c" )
    add_executable ( cholmod_l_demo "Demo/cholmod_l_demo.c" )
    add_executable ( cholmod_simple "Demo/cholmod_simple.c" )
    add_executable ( readhb         "Demo/readhb.f" )
    add_executable ( readhb2        "Demo/readhb2.f" )
    add_executable ( reade          "Demo/reade.f" )

    # Libraries required for Demo programs
    target_link_libraries ( cholmod_demo   PUBLIC cholmod )
    target_link_libraries ( cholmod_l_demo PUBLIC cholmod )
    target_link_libraries ( cholmod_simple PUBLIC cholmod )

else ( )

    message ( STATUS "Skipping the demos in CHOLMOD/Demo" )

endif ( )

