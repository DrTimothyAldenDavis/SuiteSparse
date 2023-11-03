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

set ( CHOLMOD_DATE "Oct 31, 2023" )
set ( CHOLMOD_VERSION_MAJOR 5 )
set ( CHOLMOD_VERSION_MINOR 0 )
set ( CHOLMOD_VERSION_SUB   1 )

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

include ( CheckTypeSize )
check_type_size ( "ssize_t" SSIZE_T )
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

find_package ( SuiteSparse_config 7.3.1
    PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
    find_package ( SuiteSparse_config 7.3.1 REQUIRED )
endif ( )

find_package ( COLAMD 3.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::COLAMD )
    find_package ( COLAMD 3.2.1 REQUIRED )
endif ( )

find_package ( AMD 3.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::AMD )
    find_package ( AMD 3.2.1 REQUIRED )
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

if ( SUITESPARSE_CUDA )
    # with CUDA
    add_subdirectory ( GPU )
    message ( STATUS "CUDA libraries: " ${CUDA_LIBRARIES} )
    include_directories ( GPU ${CUDAToolkit_INCLUDE_DIRS} )
    link_directories ( "GPU" "${CUDA_LIBRARIES}" "/usr/local/cuda/lib64/stubs" "/usr/local/cuda/lib64" )
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

include_directories ( Check Cholesky Utility MatrixOps Modify Partition
    Supernodal Include ${CMAKE_SOURCE_DIR} )

#-------------------------------------------------------------------------------
# dynamic cholmod library properties
#-------------------------------------------------------------------------------

file ( GLOB CHOLMOD_SOURCES "Check/cholmod_*.c" "Cholesky/cholmod_*.c"
    "Utility/cholmod_*.c" "MatrixOps/cholmod_*.c" "Modify/cholmod_*.c"
    "Partition/cholmod_*.c" "Supernodal/cholmod_*.c" )

add_library ( CHOLMOD SHARED ${CHOLMOD_SOURCES} )

set_target_properties ( CHOLMOD PROPERTIES
    VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    OUTPUT_NAME cholmod
    SOVERSION ${CHOLMOD_VERSION_MAJOR}
    PUBLIC_HEADER "Include/cholmod.h"
    WINDOWS_EXPORT_ALL_SYMBOLS ON )

if ( SUITESPARSE_CUDA )
    set_target_properties ( CHOLMOD PROPERTIES CUDA_SEPARABLE_COMPILATION on )
    set_target_properties ( CHOLMOD PROPERTIES POSITION_INDEPENDENT_CODE on )
endif ( )

target_include_directories ( CHOLMOD
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

#-------------------------------------------------------------------------------
# static cholmod library properties
#-------------------------------------------------------------------------------

if ( NOT NSTATIC )
    add_library ( CHOLMOD_static STATIC ${CHOLMOD_SOURCES} )

    set_target_properties ( CHOLMOD_static PROPERTIES
        VERSION ${CHOLMOD_VERSION_MAJOR}.${CHOLMOD_VERSION_MINOR}.${CHOLMOD_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME cholmod
        SOVERSION ${CHOLMOD_VERSION_MAJOR} )

    if ( MSVC )
        set_target_properties ( CHOLMOD_static PROPERTIES
            OUTPUT_NAME cholmod_static )
    endif ( )

    if ( SUITESPARSE_CUDA )
        set_target_properties ( CHOLMOD_static PROPERTIES CUDA_SEPARABLE_COMPILATION on )
        set_target_properties ( CHOLMOD_static PROPERTIES POSITION_INDEPENDENT_CODE on )
    endif ( )

    target_include_directories ( CHOLMOD_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::SuiteSparseConfig )
target_include_directories ( CHOLMOD PUBLIC
    "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# OpenMP:
if ( OPENMP_FOUND )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES} ")
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS} ")
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS} ")
    target_link_libraries ( CHOLMOD PRIVATE ${OpenMP_C_LIBRARIES} )
    if ( NOT NSTATIC )
        target_link_libraries ( CHOLMOD_static PUBLIC ${OpenMP_C_LIBRARIES} )
        list ( APPEND CHOLMOD_STATIC_LIBS ${OpenMP_C_LIBRARIES} )
    endif ( )
    set ( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS} " )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS} " )
    include_directories ( ${OpenMP_C_INCLUDE_DIRS} )
else ( )
    # to fix METIS: use threadprivate variables for GKRAND instead globals,
    # so multiple user threads can call cholmod_analyze in parallel on
    # different matrices, and avoid global locking of the system rand.
    include ( SuiteSparse__thread )
endif ( )

# libm:
if ( NOT WIN32 )
    target_link_libraries ( CHOLMOD PRIVATE m )
    if ( NOT NSTATIC )
        list ( APPEND CHOLMOD_STATIC_LIBS "m" )
        target_link_libraries ( CHOLMOD_static PUBLIC m )
    endif ( )
endif ( )
        list ( APPEND CHOLMOD_STATIC_LIBS "m" )

# AMD:
target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::AMD )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::AMD )
    endif ( )
endif ( )

# COLAMD:
target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::COLAMD )
if ( NOT NSTATIC )
    if ( TARGET SuiteSparse::COLAMD_static )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::COLAMD_static )
    else ( )
        target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::COLAMD )
    endif ( )
endif ( )

# BLAS and LAPACK: for the Supernodal Module
if ( NOT NSUPERNODAL )
    # LAPACK:
    message ( STATUS "LAPACK libraries:    ${LAPACK_LIBRARIES} ")
    message ( STATUS "LAPACK include:      ${LAPACK_INCLUDE_DIRS} ")
    message ( STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS} ")
    target_link_libraries ( CHOLMOD PRIVATE ${LAPACK_LIBRARIES} )
    if ( NOT NSTATIC )
        list ( APPEND CHOLMOD_STATIC_LIBS ${LAPACK_LIBRARIES} )
        target_link_libraries ( CHOLMOD_static PUBLIC ${LAPACK_LIBRARIES} )
    endif ( )
    include_directories ( ${LAPACK_INCLUDE_DIR} )

    # BLAS:
    message ( STATUS "BLAS libraries:      ${BLAS_LIBRARIES} ")
    message ( STATUS "BLAS include:        ${BLAS_INCLUDE_DIRS} ")
    message ( STATUS "BLAS linker flags:   ${BLAS_LINKER_FLAGS} ")
    target_link_libraries ( CHOLMOD PRIVATE ${BLAS_LIBRARIES} )
    if ( NOT NSTATIC )
        list ( APPEND CHOLMOD_STATIC_LIBS ${BLAS_LIBRARIES} )
        target_link_libraries ( CHOLMOD_static PUBLIC ${BLAS_LIBRARIES} )
    endif ( )
    include_directories ( ${BLAS_INCLUDE_DIRS} )
endif ( )

# CAMD and CCOLAMD:
if ( NOT NCAMD )
    target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::CAMD )
    if ( NOT NSTATIC )
        set ( CHOLMOD_STATIC_MODULES "${CHOLMOD_STATIC_MODULES} CAMD" )
        if ( TARGET SuiteSparse::CAMD_static )
            target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::CAMD_static )
        else ( )
            target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::CAMD )
        endif ( )
    endif ( )

    target_link_libraries ( CHOLMOD PRIVATE SuiteSparse::CCOLAMD )
    if ( NOT NSTATIC )
        set ( CHOLMOD_STATIC_MODULES "${CHOLMOD_STATIC_MODULES} CCOLAMD" )
        if ( TARGET SuiteSparse::CCOLAMD_static )
            target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::CCOLAMD_static )
        else ( )
            target_link_libraries ( CHOLMOD_static PUBLIC SuiteSparse::CCOLAMD )
        endif ( )
    endif ( )
endif ( )

# METIS and the Partition module:
if ( NOT NPARTITION )
    include_directories ( SuiteSparse_metis/GKlib )
    include_directories ( SuiteSparse_metis/libmetis )
    include_directories ( SuiteSparse_metis/include )
endif ( )

# CHOLMOD_CUDA
if ( SUITESPARSE_CUDA )
    target_link_libraries ( CHOLMOD PRIVATE CHOLMOD_CUDA ${CUDA_LIBRARIES} )
    set ( CHOLMOD_STATIC_MODULES "${CHOLMOD_STATIC_MODULES} CHOLMOD_CUDA" )
    set ( CHOLMOD_CFLAGS "${CHOLMOD_CFLAGS} -DSUITESPARSE_CUDA" )
    target_compile_definitions ( CHOLMOD PUBLIC "SUITESPARSE_CUDA" )
    if ( NOT NSTATIC )
        target_link_libraries ( CHOLMOD_static PUBLIC CHOLMOD_CUDA_static ${CUDA_LIBRARIES} )
        target_compile_definitions ( CHOLMOD_static PUBLIC "SUITESPARSE_CUDA" )
    endif ( )
    target_link_libraries ( CHOLMOD PRIVATE CUDA::nvrtc CUDA::cudart_static
        CUDA::nvToolsExt CUDA::cublas )
    target_include_directories ( CHOLMOD INTERFACE
        $<TARGET_PROPERTY:CUDA::cublas,INTERFACE_INCLUDE_DIRECTORIES> )
    if ( NOT NSTATIC )
        target_link_libraries ( CHOLMOD_static PUBLIC CUDA::nvrtc CUDA::cudart_static
            CUDA::nvToolsExt CUDA::cublas )
    endif ( )

    set ( old_CMAKE_EXTRA_INCLUDE_FILES CMAKE_EXTRA_INCLUDE_FILES )
    list ( APPEND CMAKE_EXTRA_INCLUDE_FILES "stdlib.h" )
    check_type_size ( "__compar_fn_t" COMPAR_FN_T )
    set ( CMAKE_EXTRA_INCLUDE_FILES old_CMAKE_EXTRA_INCLUDE_FILES )

    if ( NOT HAVE_COMPAR_FN_T )
        target_compile_definitions ( CHOLMOD PRIVATE NCOMPAR_FN_T )
        if ( NOT NSTATIC )
            target_compile_definitions ( CHOLMOD_static PRIVATE NCOMPAR_FN_T )
        endif ( )
    endif ( )

endif ( )

#-------------------------------------------------------------------------------
# CHOLMOD installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

install ( TARGETS CHOLMOD
    EXPORT CHOLMODTargets
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
if ( NOT NSTATIC )
    install ( TARGETS CHOLMOD_static
        EXPORT CHOLMODTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT CHOLMODTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT CHOLMODTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD )

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
    INSTALL_DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/CHOLMODConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CHOLMODConfigVersion.cmake
    DESTINATION ${SUITESPARSE_LIBDIR}/cmake/CHOLMOD )

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
    configure_file (
        Config/CHOLMOD.pc.in
        CHOLMOD.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CHOLMOD.pc
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
    target_link_libraries ( cholmod_demo PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig )
    if ( SUITESPARSE_CUDA )
        target_link_libraries ( cholmod_demo PUBLIC CHOLMOD_CUDA )
    endif ( )
    target_link_libraries ( cholmod_l_demo PUBLIC CHOLMOD SuiteSparse::SuiteSparseConfig )
    if ( SUITESPARSE_CUDA )
        target_link_libraries ( cholmod_l_demo PUBLIC CHOLMOD_CUDA )
    endif ( )
    target_link_libraries ( cholmod_simple PUBLIC CHOLMOD )
    if ( SUITESPARSE_CUDA )
        target_link_libraries ( cholmod_simple PUBLIC CHOLMOD_CUDA )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in CHOLMOD/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )


