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

set ( CHOLMOD_DATE "Nov 4, 2022" )
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
    ${CMAKE_SOURCE_DIR}/../COLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../AMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../CCOLAMD/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

if ( NOT DEFINED ENABLE_CUDA )
    # default is to use CUDA, if available
    set ( ENABLE_CUDA true )
endif ( )

if ( NGPL )
    # NGPL: if true, do not include any GPL-licensed module
    set ( NMATRIXOPS true )
    set ( NMODIFY true )
    set ( NSUPERNODAL true )
    set ( ENABLE_CUDA false )
    add_compile_definitions ( NGPL )
endif ( )

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

#-------------------------------------------------------------------------------
# find CHOLMOD modules
#-------------------------------------------------------------------------------

# CHOLMOD consists of a suite of modules which can be selected a la carte.  To
# not build a module, simply delete or rename the entire folder before building
# CHOLMOD, or pass the corresponding flag when using cmake:

#   -DNGL=1         if true: do not build any GPL-licensed module (MatrixOps,
#                   Modify, Supernodal, and GPU modules). See above.
#
#   -DNCHECK=1      if true: do not build the Check module.
#
#   -DNMATRIXOPS=1  if true: do not build the MatrixOps module.
#   -DNCHOLESKY=1   if true: do not build the Cholesky module.
#                   This also disables the Supernodal and Modify modules.
#   -DNMODIFY=1     if true: do not build the Modify module.
#   -DNCAMD=1       if true: do not link against CAMD and CCOLAMD.
#                   This also disables the Partition module.
#   -DNPARTITION=1  if true: do not build the Partition module.
#   -DNSUPERNODAL=1 if true: do not build the Supernodal module.

# By default, all "N..." flags are false, if not defined explicitly.

    #---------------------------------------------------------------------------
    # Check module
    #---------------------------------------------------------------------------

    if ( NOT DEFINED NCHECK )
        # by default, build the Check module
        set ( NCHECK false )
    endif ( )
    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/Check )
        # Check module does not appear so don't use it
        set ( NCHECK true )
    endif ( )
    if ( NCHECK )
        # if NCHECK is true: do not build the Check module
        add_compile_definitions ( NCHECK )
    endif ( )

    #---------------------------------------------------------------------------
    # MatrixOps module
    #---------------------------------------------------------------------------

    if ( NOT DEFINED NMATRIXOPS )
        # by default, build the MatrixOps module
        set ( NMATRIXOPS false )
    endif ( )
    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/MatrixOps )
        # MatrixOps module does not appear so don't use it
        set ( NMATRIXOPS true )
    endif ( )
    if ( NMATRIXOPS )
        # if NMATRIXOPS is true: do not build the Check module
        add_compile_definitions ( NMATRIXOPS )
    endif ( )

    #---------------------------------------------------------------------------
    # Cholesky module: requires AMD, COLAMD
    #---------------------------------------------------------------------------

    if ( NOT DEFINED NCHOLESKY )
        # by default, build the Cholesky module
        set ( NCHOLESKY false )
    endif ( )
    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/Cholesky )
        # Cholesky module does not appear so don't use it
        set ( NCHOLESKY true )
    endif ( )
    if ( NCHOLESKY )
        # if NCHOLESKY is true: do not build the Cholesky module, and do not
        # build the Supernodal or Modify modules that depend on it.
        set ( NSUPERNODAL true )
        set ( NMODIFY true )
        add_compile_definitions ( NCHOLESKY )
    endif ( )

    #---------------------------------------------------------------------------
    # Modify module: requires Cholesky
    #---------------------------------------------------------------------------

    if ( NOT DEFINED NMODIFY )
        # by default, build the Modify module
        set ( NMODIFY false )
    endif ( )
    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/Modify )
        # Modify module does not appear so don't use it
        set ( NMODIFY true )
    endif ( )
    if ( NMODIFY )
        # if NMODIFY is true: do not build the Modify module
        add_compile_definitions ( NMODIFY )
    endif ( )

    #---------------------------------------------------------------------------
    # interfaces to CAMD and CCOLAMD
    #---------------------------------------------------------------------------

    # The CHOLMOD interfaces to CAMD and CCOLAMD are in the Partition module
    # (cholmod_camd, cholmod_ccolamd, cholmod_csymamd).  The Partition module
    # has interfaces to METIS (cholmod_metis, cholmod_nesdis) that require
    # them, but the CAMD and CCOLAMD interfaces do not require METIS.  In the
    # future, the CAMD and CCOLAMD interfaces will be split into their own
    # module to make this dependency more clear.

    if ( NOT DEFINED NCAMD )
        # by default, compile with CAMD and CCOLAMD
        set ( NCAMD false )
    endif ( )

    if ( NOT NCAMD )
        # find CAMD and CCOLAMD
        find_package ( CAMD 3.0.0 )
        find_package ( CCOLAMD 3.0.0 )
        if ( NOT CAMD_FOUND OR NOT CCOLAMD_FOUND )
            # CAMD and/or CCOLAMD not found
            set ( NCAMD true )
        endif ( )
    endif ( )

    if ( NCAMD )
        # Partition module requires CAMD and CCOLAMD, so if CAMD and CCOLAMD
        # are disabled, then do not build the Partition module
        set ( NPARTITION true )
    endif ( )

    #---------------------------------------------------------------------------
    # Partition module: requires Cholesky, CAMD, CCOLAMD, and METIS
    #---------------------------------------------------------------------------

    if ( NOT DEFINED NPARTITION )
        # by default, build the Partition module
        set ( NPARTITION false )
    endif ( )
    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/Partition )
        # Partition module does not appear so don't use it.  Since this
        # folder also includes the CAMD and CCOLAMD interfaces, NCAMD must
        # be set true here as well.
        set ( NPARTITION true )
        set ( NCAMD true )
    endif ( )
    if ( NPARTITION )
        # if NPARTITION is true: do not build the Partition module
        add_compile_definitions ( NPARTITION )
    endif ( )

    if ( NCAMD )
        # if NCAMD is true: do not build the CAMD and CCOLAMD interfaces
        add_compile_definitions ( NCAMD )
    endif ( )

    #---------------------------------------------------------------------------
    # Supernodal module: requires Cholesky, BLAS, and LAPACK
    #---------------------------------------------------------------------------

    if ( NOT DEFINED NSUPERNODAL )
        # by default, build the Supernodal module
        set ( NSUPERNODAL false )
    endif ( )

    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/Supernodal )
        # Supernodal module does not appear so don't use it
        set ( NSUPERNODAL true )
    endif ( )

    if ( NSUPERNODAL )
        # if NSUPERNODAL is true: do not link against CAMD and CCOLAMD
        add_compile_definitions ( NSUPERNODAL )
    else ( )
        # if NSUPERNODAL is false: build Supernodal, which needs the BLAS
        include ( SuiteSparseBLAS )     # requires cmake 3.22
    endif ( )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

if ( SUITESPARSE_CUDA )
    # with CUDA
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DSUITESPARSE_CUDA " )
    add_subdirectory ( GPU )
    set ( CHOLMOD_CUDA cholmod_cuda ${CUDA_LIBRARIES} )
    include_directories ( GPU ${CUDAToolkit_INCLUDE_DIRS} )
    link_directories ( "GPU" "${CUDA_LIBRARIES}" "/usr/local/cuda/lib64/stubs" "/usr/local/cuda/lib64" )
else ( )
    # without CUDA and RMM
    set ( CHOLMOD_CUDA )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/cholmod.h.in" "${PROJECT_SOURCE_DIR}/Include/cholmod.h")
configure_file ( "Config/cholmod_version.tex.in" "${PROJECT_SOURCE_DIR}/Doc/cholmod_version.tex")

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Check Cholesky Core MatrixOps Modify Partition
    Supernodal Include ${CMAKE_SOURCE_DIR} ${SUITESPARSE_CONFIG_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic cholmod library properties
#-------------------------------------------------------------------------------

file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
    "Core/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
    "Partition/cholmod_*.c" "Supernodal/cholmod_*.c" )

add_library ( cholmod SHARED ${CHOLMOD_SOURCES} )

set_target_properties ( cholmod PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    SOVERSION ${CHOLMOD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cholmod.h" )

if ( SUITESPARSE_CUDA )
    set_target_properties ( cholmod PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( cholmod PROPERTIES POSITION_INDEPENDENT_CODE on )
endif ( )

#-------------------------------------------------------------------------------
# static cholmod library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
add_library ( cholmod_static STATIC ${CHOLMOD_SOURCES} )

set_target_properties ( cholmod_static PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD_REQUIRED 11
    OUTPUT_NAME cholmod
    SOVERSION ${CHOLMOD_VERSION_MAJOR} )

if ( SUITESPARSE_CUDA )
    set_target_properties ( cholmod_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( cholmod_static PROPERTIES POSITION_INDEPENDENT_CODE on )
endif ( )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# suitesparseconfig:
target_link_libraries ( cholmod PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( cholmod_static PUBLIC ${SUITESPARSE_CONFIG_LIBRARIES} )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( cholmod PUBLIC ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( cholmod_static PUBLIC ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS} " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    target_link_libraries ( cholmod PUBLIC m )
    if ( NOT NSTATIC )
    target_link_libraries ( cholmod_static PUBLIC m )
    endif ( )
endif ( )

# amd:
target_link_libraries ( cholmod PUBLIC ${AMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( cholmod_static PUBLIC ${AMD_LIBRARIES} )
endif ( )

# colamd:
target_link_libraries ( cholmod PUBLIC ${COLAMD_LIBRARIES} )
if ( NOT NSTATIC )
target_link_libraries ( cholmod_static PUBLIC ${COLAMD_LIBRARIES} )
endif ( )

# BLAS and LAPACK:
if ( NOT NSUPERNODAL )
    message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
    message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
    target_link_libraries ( cholmod PUBLIC ${BLAS_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( cholmod_static PUBLIC ${BLAS_LIBRARIES} )
    endif ( )
    include_directories ( ${BLAS_INCLUDE_DIRS} )
    message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
    message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
    target_link_libraries ( cholmod PUBLIC ${LAPACK_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( cholmod_static PUBLIC ${LAPACK_LIBRARIES} )
    endif ( )
endif ( )

# camd and ccolamd:
if ( NOT NCAMD )
    include_directories ( ${CAMD_INCLUDE_DIR} )
    target_link_libraries ( cholmod PUBLIC ${CAMD_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( cholmod_static PUBLIC ${CAMD_LIBRARIES} )
    endif ( )
    include_directories ( ${CCOLAMD_INCLUDE_DIR} )
    target_link_libraries ( cholmod PUBLIC ${CCOLAMD_LIBRARIES} )
    if ( NOT NSTATIC )
    target_link_libraries ( cholmod_static PUBLIC ${CCOLAMD_LIBRARIES} )
    endif ( )
endif ( )

# METIS and the Partition module:
if ( NOT NPARTITION )
      include_directories ( ${SUITESPARSE_METIS_INCLUDE_DIR} )
      include_directories ( SuiteSparse_metis/GKlib )
      include_directories ( SuiteSparse_metis/libmetis )
      include_directories ( SuiteSparse_metis/include )
endif ( )

# CUDA
if ( SUITESPARSE_CUDA )
    message ( STATUS "CHOLMOD cuda: " ${CHOLMOD_CUDA} )
    target_link_libraries ( cholmod PUBLIC ${CHOLMOD_CUDA} ${CUDA_LIBRARIES} )
    target_link_libraries ( cholmod_static PUBLIC ${CHOLMOD_CUDA} ${CUDA_LIBRARIES} )
endif ( )

#-------------------------------------------------------------------------------
# CHOLMOD installation location
#-------------------------------------------------------------------------------

if ( GLOBAL_INSTALL )
    # install in /usr/local/lib and /usr/local/include.
    # requires "sudo make install"
    message ( STATUS "Installation will be system-wide (requires 'sudo make install')" )
    install ( TARGETS cholmod
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
    file ( GLOB ${CMAKE_MODULES} "cmake_modules/FindCHOLMOD*.cmake" )
    install ( FILES 
        cmake_modules/FindCHOLMOD.cmake
        cmake_modules/FindCHOLMOD_CUDA.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS cholmod_static
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # also install in SuiteSparse/lib and SuiteSparse/include;
    # does not require "sudo make install", just "make install"
    message ( STATUS "Installation in ../lib and ../include only," )
    message ( STATUS "  with 'make install'. No 'sudo' required." )
    install ( TARGETS cholmod
        LIBRARY       DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
    install ( FILES 
        cmake_modules/FindCHOLMOD.cmake
        cmake_modules/FindCHOLMOD_CUDA.cmake
        DESTINATION ${SUITESPARSE_LIBDIR}/cmake/SuiteSparse 
        COMPONENT Development )
    if ( NOT NSTATIC )
    install ( TARGETS cholmod_static
        ARCHIVE       DESTINATION ${SUITESPARSE_LIBDIR} )
    endif ( )
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
    target_link_libraries ( cholmod_demo   PUBLIC cholmod ${CHOLMOD_CUDA} )
    target_link_libraries ( cholmod_l_demo PUBLIC cholmod ${CHOLMOD_CUDA} )
    target_link_libraries ( cholmod_simple PUBLIC cholmod ${CHOLMOD_CUDA} )

else ( )

    message ( STATUS "Skipping the demos in CHOLMOD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )


