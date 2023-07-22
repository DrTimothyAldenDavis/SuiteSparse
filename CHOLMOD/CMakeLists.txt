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

set ( CHOLMOD_DATE "June 16, 2023" )
set ( CHOLMOD_VERSION_MAJOR 4 )
set ( CHOLMOD_VERSION_MINOR 0 )
set ( CHOLMOD_VERSION_SUB   4 )

message ( STATUS "Building CHOLMOD version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

option ( ENABLE_CUDA "Enable CUDA acceleration" on )

if ( NGPL )
    # NGPL: if true, do not include any GPL-licensed module
    set ( NMATRIXOPS true )
    set ( NMODIFY true )
    set ( NSUPERNODAL true )
    set ( ENABLE_CUDA false )
    add_compile_definitions ( NGPL )
endif ( )

include ( SuiteSparsePolicy )
include ( SuiteSparse_ssize_t )
if ( NOT HAVE_SSIZE_T )
    # #include <sys/types.h> and ssize_t not defined (typically on Windows)
    add_compile_definitions ( NO_SSIZE_T )
endif ( )

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

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" off )
if ( NOPENMP )
    # OpenMP has been disabled
    set ( OPENMP_FOUND false )
else ( )
    find_package ( OpenMP )
endif ( )

find_package ( SuiteSparse_config 7.1.0
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.1.0 REQUIRED )
endif ( )

find_package ( COLAMD 3.0.4
    PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
if ( NOT TARGET colamd )
    find_package ( COLAMD 3.0.4 REQUIRED )
endif ( )

find_package ( AMD 3.0.4
    PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::AMD )
    find_package ( AMD 3.0.4 REQUIRED )
endif ( )

#-------------------------------------------------------------------------------
# find CHOLMOD modules
#-------------------------------------------------------------------------------

# CHOLMOD consists of a suite of modules which can be selected a la carte.  To
# not build a module, simply delete or rename the entire folder before building
# CHOLMOD, or pass the corresponding flag when using cmake:

#   -DNGPL=1        if true: do not build any GPL-licensed module (MatrixOps,
#                   Modify, Supernodal, and GPU modules). See above.
#   -DNCHECK=1      if true: do not build the Check module.
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

    option ( NCHECK "ON: do not use Check Module.  OFF (default): use Check Module" off )

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

    option ( NMATRIXOPS "ON: do not use MatrixOps Module.  OFF (default): use MatrixOps Module" off )

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

    option ( NCHOLESKY "ON: do not use Cholesky Module.  OFF (default): use Cholesky Module" off )

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

    option ( NMODIFY "ON: do not use Modify Module.  OFF (default): use Modify Module" off )

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

    option ( NCAMD "ON: do not use CAMD/CCOLAMD.  OFF (default): use CAMD/CCOLAMD" off )

    # The CHOLMOD interfaces to CAMD and CCOLAMD are in the Partition module
    # (cholmod_camd, cholmod_ccolamd, cholmod_csymamd).  The Partition module
    # has interfaces to METIS (cholmod_metis, cholmod_nesdis) that require
    # them, but the CAMD and CCOLAMD interfaces do not require METIS.  In the
    # future, the CAMD and CCOLAMD interfaces will be split into their own
    # module to make this dependency more clear.

    if ( NOT NCAMD )
        # find CAMD and CCOLAMD
        find_package ( CAMD 3.0.4
            PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::camd )
            find_package ( CAMD 3.0.4 )
        endif ( )

        find_package ( CCOLAMD 3.0.4
            PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET ccolamd )
            find_package ( CCOLAMD 3.0.4 )
        endif ( )

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

    option ( NPARTITION "ON: do not use METIS.  OFF (default): use METIS" off )

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

    option ( NSUPERNODAL "ON: do not use Supernodal Module.  OFF (default): use Supernodal Module" off )

    if ( NOT EXISTS ${CMAKE_SOURCE_DIR}/Supernodal )
        # Supernodal module does not appear so don't use it
        set ( NSUPERNODAL true )
    endif ( )

    if ( NSUPERNODAL )
        # if NSUPERNODAL is true: do not link against CAMD and CCOLAMD
        add_compile_definitions ( NSUPERNODAL )
    else ( )
        # if NSUPERNODAL is false: build Supernodal, needs BLAS and LAPACK
        include ( SuiteSparseBLAS )     # requires cmake 3.22
        include ( SuiteSparseLAPACK )   # requires cmake 3.22
    endif ( )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

add_subdirectory ( GPU )
if ( SUITESPARSE_CUDA )
    # with CUDA
    set ( CHOLMOD_CUDA cholmod_cuda ${CUDA_LIBRARIES} )
    set ( CHOLMOD_CUDA_STATIC cholmod_cuda_static ${CUDA_LIBRARIES} )
    include_directories ( GPU ${CUDAToolkit_INCLUDE_DIRS} )
    link_directories ( "GPU" "${CUDA_LIBRARIES}" "/usr/local/cuda/lib64/stubs" "/usr/local/cuda/lib64" )
else ( )
    # without CUDA and RMM
    set ( CHOLMOD_CUDA )
    set ( CHOLMOD_CUDA_STATIC )
endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/cholmod.h.in"
    "${PROJECT_SOURCE_DIR}/Include/cholmod.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/cholmod_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/cholmod_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Check Cholesky Core MatrixOps Modify Partition
    Supernodal Include ${CMAKE_SOURCE_DIR} )

#-------------------------------------------------------------------------------
# dynamic cholmod library properties
#-------------------------------------------------------------------------------

file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
    "Core/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
    "Partition/cholmod_*.c" "Supernodal/cholmod_*.c" )

add_library ( cholmod SHARED ${CHOLMOD_SOURCES} )

set_target_properties ( cholmod PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    SOVERSION ${CHOLMOD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cholmod.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

if ( SUITESPARSE_CUDA )
    set_target_properties ( cholmod PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( cholmod PROPERTIES POSITION_INDEPENDENT_CODE on )
endif ( )

target_include_directories ( cholmod
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

#-------------------------------------------------------------------------------
# static cholmod library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( cholmod_static STATIC ${CHOLMOD_SOURCES} )

    set_target_properties ( cholmod_static PROPERTIES
        VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME cholmod
        SOVERSION ${CHOLMOD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( cholmod_static PROPERTIES
            OUTPUT_NAME cholmod_static )
    endif ( )

    if ( SUITESPARSE_CUDA )
        set_target_properties ( cholmod_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( cholmod_static PROPERTIES POSITION_INDEPENDENT_CODE on )
    endif ( )

    target_include_directories ( cholmod_static
        INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>"
                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
target_link_libraries ( cholmod PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( cholmod PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( cholmod_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( cholmod_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( cholmod PRIVATE ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
        target_link_libraries ( cholmod_static PUBLIC ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS} " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( cholmod PRIVATE m )
    if ( NOT NSTATIC )
        target_link_libraries ( cholmod_static PUBLIC m )
    endif ( )
endif ( )

# AMD:
target_link_libraries ( cholmod PRIVATE SuiteSparse::AMD )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( cholmod_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( cholmod_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# colamd:
target_link_libraries ( cholmod PRIVATE colamd )
if ( NOT NSTATIC )
    if ( TARGET amd_static )
        target_link_libraries ( cholmod_static PUBLIC colamd_static )
    else ( )
        target_link_libraries ( cholmod_static PUBLIC colamd )
    endif ( )
endif ( )

# BLAS and LAPACK: for the Supernodal Module
if ( NOT NSUPERNODAL )
    # BLAS:
    message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
    message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
    message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
    target_link_libraries ( cholmod PRIVATE ${BLAS_LIBRARIES} )
    if ( NOT NSTATIC )
        target_link_libraries ( cholmod_static PUBLIC ${BLAS_LIBRARIES} )
    endif ( )
    include_directories ( ${BLAS_INCLUDE_DIRS} )

    # LAPACK:
    message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
    message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
    message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
    target_link_libraries ( cholmod PRIVATE ${LAPACK_LIBRARIES} )
    if ( NOT NSTATIC )
        target_link_libraries ( cholmod_static PUBLIC ${LAPACK_LIBRARIES} )
    endif ( )
    include_directories ( ${LAPACK_INCLUDE_DIR} )
endif ( )

# camd and ccolamd:
if ( NOT NCAMD )
    target_link_libraries ( cholmod PRIVATE SuiteSparse::camd )
    if ( NOT NSTATIC )
        if ( TARGET SuiteSparse::camd_static )
            target_link_libraries ( cholmod_static PUBLIC SuiteSparse::camd_static )
        else ( )
            target_link_libraries ( cholmod_static PUBLIC SuiteSparse::camd )
        endif ( )
    endif ( )

    target_link_libraries ( cholmod PRIVATE ccolamd )
    if ( NOT NSTATIC )
        if ( TARGET ccolamd_static )
            target_link_libraries ( cholmod_static PUBLIC ccolamd_static )
        else ( )
            target_link_libraries ( cholmod_static PUBLIC ccolamd )
        endif ( )
    endif ( )
endif ( )

# METIS and the Partition module:
if ( NOT NPARTITION )
    include_directories ( SuiteSparse_metis/GKlib )
    include_directories ( SuiteSparse_metis/libmetis )
    include_directories ( SuiteSparse_metis/include )
endif ( )

# cholmod_cuda
if ( SUITESPARSE_CUDA )
    message ( STATUS "CHOLMOD cuda: " ${CHOLMOD_CUDA} )
    target_link_libraries ( cholmod PRIVATE cholmod_cuda )
    if ( NOT NSTATIC )
        target_link_libraries ( cholmod_static PUBLIC cholmod_cuda_static )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# CHOLMOD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS cholmod
    EXPORT CHOLMODTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS cholmod_static
        EXPORT CHOLMODTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT CHOLMODTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CHOLMODTargets
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD )

configure_package_config_file (
    Config/CHOLMODConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

option ( DEMO "ON: Build the demo programs.  OFF (default): do not build the demo programs." off )
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

    if ( NOT NFORTRAN )
        add_executable ( readhb     "Demo/readhb.f" )
        add_executable ( readhb2    "Demo/readhb2.f" )
        add_executable ( reade      "Demo/reade.f" )
    endif ( )

    # Libraries required for Demo programs
    target_link_libraries ( cholmod_demo   PUBLIC cholmod ${CHOLMOD_CUDA} SuiteSparse::SuiteSparseConfig )
    target_link_libraries ( cholmod_l_demo PUBLIC cholmod ${CHOLMOD_CUDA} SuiteSparse::SuiteSparseConfig )
    target_link_libraries ( cholmod_simple PUBLIC cholmod ${CHOLMOD_CUDA} )

else ( )

    message ( STATUS "Skipping the demos in CHOLMOD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )


