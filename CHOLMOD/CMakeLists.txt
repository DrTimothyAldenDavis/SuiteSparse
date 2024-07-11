#-------------------------------------------------------------------------------
# SuiteSparse/CHOLMOD/CMakeLists.txt:  cmake for CHOLMOD
#-------------------------------------------------------------------------------

# CHOLMOD: Copyright (c) 2005-2023, Timothy A. Davis.
# Copyright and license varies by module.

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

# cmake 3.22 is required to find the BLAS/LAPACK
cmake_minimum_required ( VERSION 3.22 )

set ( CHOLMOD_DATE "June 20, 2024" )
set ( CHOLMOD_VERSION_MAJOR 5 CACHE STRING "" FORCE )
set ( CHOLMOD_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( CHOLMOD_VERSION_SUB   0 CACHE STRING "" FORCE )

message ( STATUS "Building CHOLMOD version: v"
    ${CHOLMOD_VERSION_MAJOR}.
    ${CHOLMOD_VERSION_MINOR}.
    ${CHOLMOD_VERSION_SUB} " (" ${CHOLMOD_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( CHOLMOD
        VERSION "${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}"
        LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

option ( CHOLMOD_USE_CUDA "ON (default): enable CUDA acceleration for CHOLMOD, OFF: do not use CUDA" ${SUITESPARSE_USE_CUDA} )
if ( NOT SUITESPARSE_USE_CUDA )
    set ( CHOLMOD_USE_CUDA "OFF" CACHE STRING "" FORCE )
endif ( )

option ( CHOLMOD_GPL "ON (default): enable GPL-licensed modules, OFF: do not use any CHOLMOD GPL-licensed modules" ON )
if ( NOT CHOLMOD_GPL )
    # CHOLMOD_GPL: if OFF, do not include any GPL-licensed module
    set ( CHOLMOD_MATRIXOPS OFF )
    set ( CHOLMOD_MODIFY OFF )
    set ( CHOLMOD_SUPERNODAL OFF )
    set ( CHOLMOD_USE_CUDA "OFF" CACHE STRING "" FORCE ) # OK: disabled by user-setting of CHOLMOD_GPL
    add_compile_definitions ( NGPL )
endif ( )

include ( CheckTypeSize )
check_type_size ( "ssize_t" SSIZE_T )
if ( NOT HAVE_SSIZE_T )
    # #include <sys/types.h> and ssize_t not defined (typically on Windows)
    add_compile_definitions ( NO_SSIZE_T )
endif ( )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

if ( SUITESPARSE_HAS_CUDA AND CHOLMOD_USE_CUDA )
    # CHOLMOD with CUDA
    enable_language ( CUDA CXX )
    set ( CHOLMOD_HAS_CUDA ON )
else ( )
    set ( CHOLMOD_HAS_CUDA OFF )
endif ( )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND CHOLMOD_USE_CUDA AND NOT CHOLMOD_HAS_CUDA )
    message ( FATAL_ERROR "CUDA required for CHOLMOD but not found" )
endif ( )

#-------------------------------------------------------------------------------
# find OpenMP
#-------------------------------------------------------------------------------

option ( CHOLMOD_USE_OPENMP "ON: Use OpenMP in CHOLMOD if available.  OFF: Do not use OpenMP.  (Default: SUITESPARSE_USE_OPENMP)" ${SUITESPARSE_USE_OPENMP} )
if ( CHOLMOD_USE_OPENMP )
    if ( CMAKE_VERSION VERSION_LESS 3.24 )
        find_package ( OpenMP COMPONENTS C )
    else ( )
        find_package ( OpenMP COMPONENTS C GLOBAL )
    endif ( )
else ( )
    # OpenMP has been disabled
    set ( OpenMP_C_FOUND OFF )
endif ( )

if ( CHOLMOD_USE_OPENMP AND OpenMP_C_FOUND )
    set ( CHOLMOD_HAS_OPENMP ON )
else ( )
    set ( CHOLMOD_HAS_OPENMP OFF )
endif ( )
message ( STATUS "CHOLMOD has OpenMP: ${CHOLMOD_HAS_OPENMP}" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND CHOLMOD_USE_OPENMP AND NOT CHOLMOD_HAS_OPENMP )
    message ( FATAL_ERROR "OpenMP required for CHOLMOD but not found" )
endif ( )

#-------------------------------------------------------------------------------
# find other library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.8.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.8.0 REQUIRED )
    endif ( )

    find_package ( AMD 3.3.2
        PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::AMD )
        find_package ( AMD 3.3.2 REQUIRED )
    endif ( )

    find_package ( COLAMD 3.3.3
        PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::COLAMD )
        find_package ( COLAMD 3.3.3 REQUIRED )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# find CHOLMOD modules
#-------------------------------------------------------------------------------

# CHOLMOD consists of a suite of modules which can be selected a la carte.  To
# not build a module, simply delete or rename the entire folder before building
# CHOLMOD, or pass the corresponding flag when using cmake:

#   -DCHOLMOD_GPL=0        if OFF: do not build any GPL-licensed module
#                          (MatrixOps, Modify, Supernodal, and GPU modules).
#   -DCHOLMOD_CHECK=0      if OFF: do not build the Check module.
#   -DCHOLMOD_MATRIXOPS=0  if OFF: do not build the MatrixOps module.
#   -DCHOLMOD_CHOLESKY=0   if OFF: do not build the Cholesky module.
#                          This also disables the Supernodal and Modify modules.
#   -DCHOLMOD_MODIFY=0     if OFF: do not build the Modify module.
#   -DCHOLMOD_CAMD=0       if OFF: do not link against CAMD and CCOLAMD.
#                          This also disables the Partition module.
#   -DCHOLMOD_PARTITION=0  if OFF: do not build the Partition module.
#   -DCHOLMOD_SUPERNODAL=0 if OFF: do not build the Supernodal module.

# By default, all the above flags are ON, if not defined explicitly.

    #---------------------------------------------------------------------------
    # Check module
    #---------------------------------------------------------------------------

    option ( CHOLMOD_CHECK "ON (default): use Check Module.  OFF: do not use Check Module" ON )

    if ( NOT EXISTS ${PROJECT_SOURCE_DIR}/Check )
        # Check module does not appear so don't use it
        set ( CHOLMOD_CHECK OFF )
    endif ( )
    if ( NOT CHOLMOD_CHECK )
        # if CHOLMOD_CHECK is OFF: do not build the Check module
        add_compile_definitions ( NCHECK )
    endif ( )

    #---------------------------------------------------------------------------
    # MatrixOps module
    #---------------------------------------------------------------------------

    option ( CHOLMOD_MATRIXOPS "ON (default): use MatrixOps Module.  OFF: do not use MatrixOps Module" ON )

    if ( NOT EXISTS ${PROJECT_SOURCE_DIR}/MatrixOps )
        # MatrixOps module does not appear so don't use it
        set ( CHOLMOD_MATRIXOPS OFF )
    endif ( )
    if ( NOT CHOLMOD_MATRIXOPS )
        # if CHOLMOD_MATRIXOPS is OFF: do not build the Check module
        add_compile_definitions ( NMATRIXOPS )
    endif ( )

    #---------------------------------------------------------------------------
    # Cholesky module: requires AMD, COLAMD
    #---------------------------------------------------------------------------

    option ( CHOLMOD_CHOLESKY "ON (default): use Cholesky Module.  OFF: do not use Cholesky Module" ON )

    if ( NOT EXISTS ${PROJECT_SOURCE_DIR}/Cholesky )
        # Cholesky module does not appear so don't use it
        set ( CHOLMOD_CHOLESKY OFF )
    endif ( )
    if ( NOT CHOLMOD_CHOLESKY )
        # if CHOLMOD_CHOLESKY is OFF: do not build the Cholesky module, and do not
        # build the Supernodal or Modify modules that depend on it.
        set ( CHOLMOD_SUPERNODAL OFF )
        set ( CHOLMOD_MODIFY OFF )
        add_compile_definitions ( NCHOLESKY )
    endif ( )

    #---------------------------------------------------------------------------
    # Modify module: requires Cholesky
    #---------------------------------------------------------------------------

    option ( CHOLMOD_MODIFY "ON (default): use Modify Module.  OFF: do not use Modify Module" ON )

    if ( NOT EXISTS ${PROJECT_SOURCE_DIR}/Modify )
        # Modify module does not appear so don't use it
        set ( CHOLMOD_MODIFY OFF )
    endif ( )
    if ( NOT CHOLMOD_MODIFY )
        # if CHOLMOD_MODIFY is OFF: do not build the Modify module
        add_compile_definitions ( NMODIFY )
    endif ( )

    #---------------------------------------------------------------------------
    # interfaces to CAMD and CCOLAMD
    #---------------------------------------------------------------------------

    option ( CHOLMOD_CAMD "ON (default): use CAMD/CCOLAMD.  OFF: do not use CAMD/CCOLAMD" ON )

    # The CHOLMOD interfaces to CAMD and CCOLAMD are in the Partition module
    # (cholmod_camd, cholmod_ccolamd, cholmod_csymamd).  The Partition module
    # has interfaces to METIS (cholmod_metis, cholmod_nesdis) that require
    # them, but the CAMD and CCOLAMD interfaces do not require METIS.  In the
    # future, the CAMD and CCOLAMD interfaces will be split into their own
    # module to make this dependency more clear.

    if ( CHOLMOD_CAMD AND NOT SUITESPARSE_ROOT_CMAKELISTS )
        # find CAMD and CCOLAMD
        find_package ( CAMD 3.3.2
            PATHS ${CMAKE_SOURCE_DIR}/../CAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::CAMD )
            find_package ( CAMD 3.3.2 )
        endif ( )

        find_package ( CCOLAMD 3.3.3
            PATHS ${CMAKE_SOURCE_DIR}/../CCOLAMD/build NO_DEFAULT_PATH )
        if ( NOT TARGET SuiteSparse::CCOLAMD )
            find_package ( CCOLAMD 3.3.3 )
        endif ( )

        if ( NOT CAMD_FOUND OR NOT CCOLAMD_FOUND )
            # CAMD and/or CCOLAMD not found
            set ( CHOLMOD_CAMD OFF )
        endif ( )
    endif ( )

    if ( NOT CHOLMOD_CAMD )
        # Partition module requires CAMD and CCOLAMD, so if CAMD and CCOLAMD
        # are disabled, then do not build the Partition module
        set ( CHOLMOD_PARTITION OFF )
    endif ( )

    #---------------------------------------------------------------------------
    # Partition module: requires Cholesky, CAMD, CCOLAMD, and METIS
    #---------------------------------------------------------------------------

    option ( CHOLMOD_PARTITION "ON (default): use METIS.  OFF: do not use METIS" ON )

    if ( NOT EXISTS ${PROJECT_SOURCE_DIR}/Partition )
        # Partition module does not appear so don't use it.  Since this folder
        # also includes the CAMD and CCOLAMD interfaces, CHOLMOD_CAMD must be
        # set OFF here as well.
        set ( CHOLMOD_PARTITION OFF )
        set ( CHOLMOD_CAMD OFF )
    endif ( )
    if ( NOT CHOLMOD_PARTITION )
        # if CHOLMOD_PARTITION is OFF: do not build the Partition module
        add_compile_definitions ( NPARTITION )
    endif ( )

    if ( NOT CHOLMOD_CAMD )
        # if CHOLMOD_CAMD is OFF: do not build the CAMD and CCOLAMD interfaces
        add_compile_definitions ( NCAMD )
    endif ( )

    #---------------------------------------------------------------------------
    # Supernodal module: requires Cholesky, BLAS, and LAPACK
    #---------------------------------------------------------------------------

    option ( CHOLMOD_SUPERNODAL "ON (default): use Supernodal Module.  OFF: do not use Supernodal Module" ON )

    if ( NOT EXISTS ${PROJECT_SOURCE_DIR}/Supernodal )
        # Supernodal module does not appear so don't use it
        set ( CHOLMOD_SUPERNODAL OFF )
    endif ( )

    if ( NOT CHOLMOD_SUPERNODAL )
        # if CHOLMOD_SUPERNODAL is OFF: do not build Supernodal module
        add_compile_definitions ( NSUPERNODAL )
    else ( )
        # if CHOLMOD_SUPERNODAL is ON: build Supernodal, needs BLAS and LAPACK
        include ( SuiteSparseBLAS )     # requires cmake 3.22
        include ( SuiteSparseLAPACK )   # requires cmake 3.22
    endif ( )

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

if ( CHOLMOD_CHECK )
    set ( CHOLMOD_HAS_CHECK ON )
else ( )
    set ( CHOLMOD_HAS_CHECK OFF )
endif ( )

if ( CHOLMOD_CHOLESKY )
    set ( CHOLMOD_HAS_CHOLESKY ON )
else ( )
    set ( CHOLMOD_HAS_CHOLESKY OFF )
endif ( )

if ( CHOLMOD_PARTITION )
    set ( CHOLMOD_HAS_PARTITION ON )
else ( )
    set ( CHOLMOD_HAS_PARTITION OFF )
endif ( )

if ( CHOLMOD_CAMD )
    # CHOLMOD has both CAMD and CCOLAMD
    set ( CHOLMOD_HAS_CAMD ON )
else ( )
    set ( CHOLMOD_HAS_CAMD OFF )
endif ( )

if ( CHOLMOD_MATRIXOPS )
    set ( CHOLMOD_HAS_MATRIXOPS ON )
else ( )
    set ( CHOLMOD_HAS_MATRIXOPS OFF )
endif ( )

if ( CHOLMOD_MODIFY )
    set ( CHOLMOD_HAS_MODIFY ON )
else ( )
    set ( CHOLMOD_HAS_MODIFY OFF )
endif ( )

if ( CHOLMOD_SUPERNODAL )
    set ( CHOLMOD_HAS_SUPERNODAL ON )
else ( )
    set ( CHOLMOD_HAS_SUPERNODAL OFF )
endif ( )

if ( CHOLMOD_GPL )
    set ( CHOLMOD_HAS_GPL ON )
else ( )
    set ( CHOLMOD_HAS_GPL OFF )
endif ( )

configure_file ( "Config/cholmod.h.in"
    "${PROJECT_SOURCE_DIR}/Include/cholmod.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/cholmod_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/cholmod_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Check Cholesky Utility MatrixOps Modify Partition
    Supernodal Include ${PROJECT_SOURCE_DIR} )

#-------------------------------------------------------------------------------
# dynamic cholmod library properties
#-------------------------------------------------------------------------------

file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
    "Utility/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
    "Partition/cholmod_*.c" "Supernodal/cholmod_*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( CHOLMOD SHARED ${CHOLMOD_SOURCES} )

    set_target_properties ( CHOLMOD PROPERTIES
        VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME cholmod
        SOVERSION ${CHOLMOD_VERSION_MAJOR}
        PUBLIC_HEADER "Include/cholmod.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        LINKER_LANGUAGE C )

    if ( CHOLMOD_HAS_CUDA )
        set_target_properties ( CHOLMOD PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
        set_target_properties ( CHOLMOD PROPERTIES POSITION_INDEPENDENT_CODE ON )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( CHOLMOD PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( CHOLMOD
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static cholmod library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( CHOLMOD_static STATIC ${CHOLMOD_SOURCES} )

    set_target_properties ( CHOLMOD_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME cholmod
        PUBLIC_HEADER "Include/cholmod.h"
        LINKER_LANGUAGE C )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( CHOLMOD_static PROPERTIES
            OUTPUT_NAME cholmod_static )
    endif ( )

    if ( CHOLMOD_HAS_CUDA )
        set_target_properties ( CHOLMOD_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( CHOLMOD_static PROPERTIES POSITION_INDEPENDENT_CODE on )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( CHOLMOD_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( CHOLMOD_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( CHOLMOD PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::AMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::AMD )
    endif ( )
endif ( )

# COLAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::COLAMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::COLAMD_static )
        target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::COLAMD_static )
    else ( )
        target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::COLAMD )
    endif ( )
endif ( )

# CAMD and CCOLAMD:
if ( CHOLMOD_CAMD )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::CAMD )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( CHOLMOD_STATIC_MODULES "${CHOLMOD_STATIC_MODULES} CAMD" )
        if ( TARGET SuiteSparse::CAMD_static )
            target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::CAMD_static )
        else ( )
            target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::CAMD )
        endif ( )
    endif ( )

    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::CCOLAMD )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        set ( CHOLMOD_STATIC_MODULES "${CHOLMOD_STATIC_MODULES} CCOLAMD" )
        if ( TARGET SuiteSparse::CCOLAMD_static )
            target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::CCOLAMD_static )
        else ( )
            target_link_libraries ( CHOLMOD_static PRIVATE SuiteSparse::CCOLAMD )
        endif ( )
    endif ( )
endif ( )

# OpenMP:
if ( CHOLMOD_HAS_OPENMP )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES}" )
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS}" )
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS}" )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE OpenMP::OpenMP_C )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( CHOLMOD_static PRIVATE OpenMP::OpenMP_C )
        list ( APPEND CHOLMOD_STATIC_LIBS ${OpenMP_C_LIBRARIES} )
    endif ( )
else ( )
    # to fix METIS: use threadprivate variables for GKRAND instead globals,
    # so multiple user threads can call cholmod_analyze in parallel on
    # different matrices, and avoid global locking of the system rand.
    include ( SuiteSparse__thread )
endif ( )

# libm:
include ( CheckSymbolExists )
check_symbol_exists ( fmax "math.h" NO_LIBM )
if ( NOT NO_LIBM )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND CHOLMOD_STATIC_LIBS "m" )
        target_link_libraries ( CHOLMOD_static PRIVATE m )
    endif ( )
endif ( )

# BLAS and LAPACK: for the Supernodal Module
if ( CHOLMOD_SUPERNODAL )
    # LAPACK:
    message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES}" )
    message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS}" )
    message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS}" )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE ${LAPACK_LIBRARIES} )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND CHOLMOD_STATIC_LIBS ${LAPACK_LIBRARIES} )
        target_link_libraries ( CHOLMOD_static PRIVATE ${LAPACK_LIBRARIES} )
    endif ( )
    include_directories ( ${LAPACK_INCLUDE_DIR} )

    # BLAS:
    message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES}" )
    message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS}" )
    message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS}" )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE ${BLAS_LIBRARIES} )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND CHOLMOD_STATIC_LIBS ${BLAS_LIBRARIES} )
        target_link_libraries ( CHOLMOD_static PRIVATE ${BLAS_LIBRARIES} )
    endif ( )
    include_directories ( ${BLAS_INCLUDE_DIRS} )
endif ( )

# METIS and the Partition module:
if ( CHOLMOD_PARTITION )
    include_directories ( SuiteSparse_metis/GKlib )
    include_directories ( SuiteSparse_metis/libmetis )
    include_directories ( SuiteSparse_metis/include )
endif ( )

# CHOLMOD_CUDA
if ( CHOLMOD_HAS_CUDA )
#   cmake now configures cholmod.h with "#define CHOLMOD_HAS_CUDA" if CHOLMOD is
#   being compiled with CUDA, so the -DCHOLMOD_HAS_CUDA flag is no longer needed.
#   if ( BUILD_SHARED_LIBS )
#       target_compile_definitions ( CHOLMOD PUBLIC "CHOLMOD_HAS_CUDA" )
#   endif ( )
#   set ( CHOLMOD_CFLAGS "${CHOLMOD_CFLAGS} -DCHOLMOD_HAS_CUDA" )
#   if ( BUILD_STATIC_LIBS )
#       target_compile_definitions ( CHOLMOD_static PUBLIC "CHOLMOD_HAS_CUDA" )
#   endif ( )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( CHOLMOD PRIVATE CUDA::nvrtc CUDA::cudart_static CUDA::cublas )
        target_include_directories ( CHOLMOD INTERFACE
            $<TARGET_PROPERTY:CUDA::cublas,INTERFACE_INCLUDE_DIRECTORIES> )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( CHOLMOD_static PRIVATE CUDA::nvrtc CUDA::cudart_static )
        # FIXME: Ok to hardcode CUDA library names like this?
        list ( APPEND CHOLMOD_STATIC_LIBS "-L${CUDAToolkit_LIBRARY_DIR};-lcuda;-lcudart_static" )
        if ( TARGET CUDA::cublas_static )
            target_link_libraries ( CHOLMOD_static PRIVATE CUDA::cublas_static )
            target_include_directories ( CHOLMOD_static INTERFACE
                $<TARGET_PROPERTY:CUDA::cublas_static,INTERFACE_INCLUDE_DIRECTORIES> )
            list ( APPEND CHOLMOD_STATIC_LIBS "-lcublas_static;-lcublasLt_static;-lculibos" )
        else ( )
            target_link_libraries ( CHOLMOD_static PRIVATE CUDA::cublas )
            target_include_directories ( CHOLMOD_static INTERFACE
                $<TARGET_PROPERTY:CUDA::cublas,INTERFACE_INCLUDE_DIRECTORIES> )
            list ( APPEND CHOLMOD_STATIC_LIBS "-lcublas" )
        endif ( )
    endif ( )

    set ( _orig_CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_EXTRA_INCLUDE_FILES} )
    list ( APPEND CMAKE_EXTRA_INCLUDE_FILES "stdlib.h" )
    check_type_size ( "__compar_fn_t" COMPAR_FN_T )
    set ( CMAKE_EXTRA_INCLUDE_FILES ${_orig_CMAKE_EXTRA_INCLUDE_FILES} )

    if ( NOT HAVE_COMPAR_FN_T )
        if ( BUILD_SHARED_LIBS )
            target_compile_definitions ( CHOLMOD PRIVATE NCOMPAR_FN_T )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_compile_definitions ( CHOLMOD_static PRIVATE NCOMPAR_FN_T )
        endif ( )
    endif ( )

endif ( )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

if ( CHOLMOD_HAS_CUDA )
    # with CUDA
    add_subdirectory ( GPU )
    message ( STATUS "CUDA libraries: " ${CUDA_LIBRARIES} )
endif ( )

#-------------------------------------------------------------------------------
# CHOLMOD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS CHOLMOD
        EXPORT CHOLMODTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS CHOLMOD_static
        EXPORT CHOLMODTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT CHOLMODTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CHOLMODTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/CHOLMOD )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE on )
configure_package_config_file (
    Config/CHOLMODConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE off )
configure_package_config_file (
    Config/CHOLMODConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/CHOLMODConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/CHOLMOD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/CHOLMODConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/CHOLMOD )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( CHOLMOD_STATIC_LIBS_LIST ${CHOLMOD_STATIC_LIBS} )
    set ( CHOLMOD_STATIC_LIBS "" )
    foreach ( _lib ${CHOLMOD_STATIC_LIBS_LIST} )
        if ( ${_lib} MATCHES "-[lL].*" )
            # take -L or -l flags as is
            set ( CHOLMOD_STATIC_LIBS "${CHOLMOD_STATIC_LIBS} ${_lib}" )
            continue ()
        endif ( )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( CHOLMOD_STATIC_LIBS "${CHOLMOD_STATIC_LIBS} -l${_lib}" )
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
                    set ( CHOLMOD_STATIC_LIBS "${CHOLMOD_STATIC_LIBS} -l${_libname}" )
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
    if ( BUILD_SHARED_LIBS )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:CHOLMOD> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:CHOLMOD_static> )
    endif ( )
    configure_file (
        Config/CHOLMOD.pc.in
        CHOLMOD.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT CHOLMOD.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/CHOLMOD.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CHOLMOD.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    add_executable ( cholmod_di_simple "Demo/cholmod_di_simple.c" )
    add_executable ( cholmod_dl_simple "Demo/cholmod_dl_simple.c" )
    add_executable ( cholmod_si_simple "Demo/cholmod_si_simple.c" )
    add_executable ( cholmod_sl_simple "Demo/cholmod_sl_simple.c" )

    if ( SUITESPARSE_HAS_FORTRAN )
        add_executable ( readhb     "Demo/readhb.f" )
        add_executable ( readhb2    "Demo/readhb2.f" )
        add_executable ( reade      "Demo/reade.f" )
    endif ( )

    if ( BUILD_SHARED_LIBS )

        # link the demos with shared libraries
        target_link_libraries ( cholmod_di_simple PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( cholmod_si_simple PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( cholmod_dl_simple PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( cholmod_sl_simple PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig )

    else ( )

        # link the demos with static libraries
        target_link_libraries ( cholmod_di_simple PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( cholmod_si_simple PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( cholmod_dl_simple PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig )
        target_link_libraries ( cholmod_sl_simple PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig )

    endif ( )

endif ( )

#-------------------------------------------------------------------------------
# testing
#-------------------------------------------------------------------------------

include ( CTest )

if ( SUITESPARSE_DEMOS OR BUILD_TESTING )

    add_executable ( cholmod_di_demo "Demo/cholmod_di_demo.c" )
    add_executable ( cholmod_dl_demo "Demo/cholmod_dl_demo.c" )
    add_executable ( cholmod_si_demo "Demo/cholmod_si_demo.c" )
    add_executable ( cholmod_sl_demo "Demo/cholmod_sl_demo.c" )

    # Libraries required for tests
    if ( BUILD_SHARED_LIBS )

        # link the tests with shared libraries
        target_link_libraries ( cholmod_di_demo PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( cholmod_si_demo PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( cholmod_dl_demo PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( cholmod_sl_demo PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )

    else ( )

        # link the tests with static libraries
        target_link_libraries ( cholmod_di_demo PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( cholmod_si_demo PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( cholmod_dl_demo PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( cholmod_sl_demo PUBLIC CHOLMOD_static SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )

    endif ( )

    if ( CHOLMOD_CAMD )

        target_link_libraries ( cholmod_di_demo PUBLIC SuiteSparse::CAMD SuiteSparse::CCOLAMD )
        target_link_libraries ( cholmod_si_demo PUBLIC SuiteSparse::CAMD SuiteSparse::CCOLAMD )
        target_link_libraries ( cholmod_dl_demo PUBLIC SuiteSparse::CAMD SuiteSparse::CCOLAMD )
        target_link_libraries ( cholmod_sl_demo PUBLIC SuiteSparse::CAMD SuiteSparse::CCOLAMD )

    endif ( )

    add_test ( NAME CHOLMOD_int32_double_bcsstk01
        COMMAND cholmod_di_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk01.tri )
    add_test ( NAME CHOLMOD_int64_double_bcsstk01
        COMMAND cholmod_dl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk01.tri )
    add_test ( NAME CHOLMOD_int32_single_bcsstk01
        COMMAND cholmod_si_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk01.tri )
    add_test ( NAME CHOLMOD_int64_single_bcsstk01
        COMMAND cholmod_sl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk01.tri )

    add_test ( NAME CHOLMOD_int32_double_lp_afiro
        COMMAND cholmod_di_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/lp_afiro.tri )
    add_test ( NAME CHOLMOD_int64_double_lp_afiro
        COMMAND cholmod_dl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/lp_afiro.tri )
    add_test ( NAME CHOLMOD_int32_single_lp_afiro
        COMMAND cholmod_si_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/lp_afiro.tri )
    add_test ( NAME CHOLMOD_int64_single_lp_afiro
        COMMAND cholmod_sl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/lp_afiro.tri )

    add_test ( NAME CHOLMOD_int32_double_can24
        COMMAND cholmod_di_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/can___24.mtx )
    add_test ( NAME CHOLMOD_int64_double_can24
        COMMAND cholmod_dl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/can___24.mtx )
    add_test ( NAME CHOLMOD_int32_single_can24
        COMMAND cholmod_si_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/can___24.mtx )
    add_test ( NAME CHOLMOD_int64_single_can24
        COMMAND cholmod_sl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/can___24.mtx )

    add_test ( NAME CHOLMOD_int32_double_complex
        COMMAND cholmod_di_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/c.tri )
    add_test ( NAME CHOLMOD_int64_double_complex
        COMMAND cholmod_dl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/c.tri )
    add_test ( NAME CHOLMOD_int32_single_complex
        COMMAND cholmod_si_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/c.tri )
    add_test ( NAME CHOLMOD_int64_single_complex
        COMMAND cholmod_sl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/c.tri )

    add_test ( NAME CHOLMOD_int32_double_supernodal
        COMMAND cholmod_di_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk02.tri )
    add_test ( NAME CHOLMOD_int64_double_supernodal
        COMMAND cholmod_dl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk02.tri )
    add_test ( NAME CHOLMOD_int32_single_supernodal
        COMMAND cholmod_si_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk02.tri )
    add_test ( NAME CHOLMOD_int64_single_supernodal
        COMMAND cholmod_sl_demo ${PROJECT_SOURCE_DIR}/Demo/Matrix/bcsstk02.tri )

    if ( WIN32 AND BUILD_SHARED_LIBS )
        # Set PATH to pick up the necessary libraries for all tests

        set ( CHOLMOD_CTEST_NAMES
            CHOLMOD_int32_double_bcsstk01 CHOLMOD_int64_double_bcsstk01 CHOLMOD_int32_single_bcsstk01 CHOLMOD_int64_single_bcsstk01
            CHOLMOD_int32_double_lp_afiro CHOLMOD_int64_double_lp_afiro CHOLMOD_int32_single_lp_afiro CHOLMOD_int64_single_lp_afiro
            CHOLMOD_int32_double_can24 CHOLMOD_int64_double_can24 CHOLMOD_int32_single_can24 CHOLMOD_int64_single_can24
            CHOLMOD_int32_double_complex CHOLMOD_int64_double_complex CHOLMOD_int32_single_complex CHOLMOD_int64_single_complex
            CHOLMOD_int32_double_supernodal CHOLMOD_int64_double_supernodal CHOLMOD_int32_single_supernodal CHOLMOD_int64_single_supernodal )

        set ( CHOLMOD_DEPENDENCIES
            CHOLMOD SuiteSparse::SuiteSparseConfig SuiteSparse::AMD SuiteSparse::COLAMD )
        if ( CHOLMOD_CAMD )
            list ( APPEND CHOLMOD_DEPENDENCIES SuiteSparse::CAMD SuiteSparse::CCOLAMD )
        endif ( )
        set ( CHOLMOD_CTEST_PATH_MODIFICATION "" )
        foreach ( cholmod_dependency ${CHOLMOD_DEPENDENCIES} )
            list ( APPEND CHOLMOD_CTEST_PATH_MODIFICATION PATH=path_list_prepend:$<TARGET_FILE_DIR:${cholmod_dependency}> )
        endforeach ( )
        foreach( cholmod_ctest ${CHOLMOD_CTEST_NAMES} )
            set_tests_properties ( ${cholmod_ctest} PROPERTIES
                ENVIRONMENT_MODIFICATION "${CHOLMOD_CTEST_PATH_MODIFICATION}" )
        endforeach ( )

    endif ( )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )
