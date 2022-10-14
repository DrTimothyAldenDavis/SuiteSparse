#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/CMakeLists.txt:  cmake for CHOLMOD
#-------------------------------------------------------------------------------

# CHOLMOD: Copyright (c) 2005-2022, Timothy A. Davis.
# Copyright and license varies by module.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.22 )

set ( CHOLMOD_DATE "Oct 31, 2022" )  # FIXME
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
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules
    )

# FIXME: CUDA doesn't work yet with this cmake script
set ( ENABLE_CUDA false )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    # CHOLMOD with CUDA
    project ( cholmod
            VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
            LANGUAGES CUDA C CXX )
else ( )
    # CHOLMOD without CUDA
    project ( cholmod
            VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
            LANGUAGES C )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

enable_language ( C Fortran )
find_package ( OpenMP )
find_package ( SuiteSparse_config 6.0.0 REQUIRED )
find_package ( COLAMD 3.0.0 REQUIRED )
find_package ( AMD 3.0.0 REQUIRED )
find_package ( CAMD 3.0.0 )
find_package ( CCOLAMD 3.0.0 )
find_package ( SuiteSparse_metis 5.1.0 )

include ( SuiteSparseBLAS )     # requires cmake 3.22

if ( NOT SUITESPARSE_METIS_FOUND )
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DNPARTITION " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNPARTITION " )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

# determine which modules of CHOLMOD to include in the library
if ( NGPL )
    add_compile_definitions ( NGPL )
endif ( )

configure_file ( "Config/cholmod.h.in" "${PROJECT_SOURCE_DIR}/Include/cholmod.h")
configure_file ( "Config/cholmod_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/cholmod_version.tex")

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Check Cholesky Core MatrixOps Modify Partition
    Supernodal Include ${SUITESPARSE_CONFIG_INCLUDE_DIR} ${AMD_INCLUDE_DIR}
    ${COLAMD_INCLUDE_DIR} ${CAMD_INCLUDE_DIR} ${CCOLAMD_INCLUDE_DIR}
    ${SUITESPARSE_METIS_INCLUDE_DIR} )

if ( SUITESPARSE_CUDA )
    include_directories ( GPU ${CUDAToolkit_INCLUDE_DIRS} )
endif ( )

#-------------------------------------------------------------------------------
# dynamic cholmod library properties
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
        "Core/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
        "Partition/cholmod_*.c" "Supernodal/cholmod_*.c" "GPU/cholmod_*.c"
        "GPU/cholmod_*.cu" )
else ( )
    file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
        "Core/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
        "Partition/cholmod_*.c" "Supernodal/cholmod_*.c" )
endif ( )

add_library ( cholmod SHARED ${CHOLMOD_SOURCES} )

SET_TARGET_PROPERTIES ( cholmod PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CHOLMOD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cholmod.h" )

if ( SUITESPARSE_CUDA )
    set_target_properties ( cholmod PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( cholmod PROPERTIES POSITION_INDEPENDENT_CODE on )
    set_target_properties ( cholmod PROPERTIES CUDA_ARCHITECTURES "75" )
endif ( )

#-------------------------------------------------------------------------------
# static cholmod library properties
#-------------------------------------------------------------------------------

add_library ( cholmod_static STATIC ${CHOLMOD_SOURCES} )

SET_TARGET_PROPERTIES ( cholmod_static PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME cholmod
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

if ( SUITESPARSE_CUDA )
    set_target_properties ( cholmod_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( cholmod_static PROPERTIES POSITION_INDEPENDENT_CODE on )
    set_target_properties ( cholmod_static PROPERTIES CUDA_ARCHITECTURES "75" )
endif ( )

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
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS} " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} " )
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
include_directories ( ${BLAS_INCLUDE_DIRS} )

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
if ( SUITESPARSE_METIS_FOUND )
    target_link_libraries ( cholmod PUBLIC ${SUITESPARSE_METIS_LIBRARIES} )
    target_link_libraries ( cholmod_static PUBLIC ${SUITESPARSE_METIS_LIBRARIES} )
endif ( )

# CUDA: cudart and cublas libraries
if ( SUITESPARSE_CUDA )
    link_directories ( ${CUDA_LIBRARIES} ${CUDAToolkit_LIBRARY_DIR} )
    target_link_libraries ( cholmod PUBLIC ${CUDA_LIBRARIES}
        CUDA::cudart CUDA::cublas CUDA::nvrtc CUDA::nvToolsExt )
    target_link_libraries ( cholmod_static PUBLIC ${CUDA_LIBRARIES}
        CUDA::cudart_static CUDA::cublas CUDA::nvrtc CUDA::nvToolsExt )
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DGPU_BLAS " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_BLAS --std=c++17 -fPIC" )
endif ( )

#-------------------------------------------------------------------------------
# CHOLMOD installation location
#---------------------------------------------------------------Z--------------

# use "cmake -DSUITESPARSE_LOCAL=1" to install only in SuiteSparse/lib and
# SuiteSparse/include.  

if ( NOT SUITESPARSE_LOCAL )
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
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( TARGETS cholmod_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
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

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )


