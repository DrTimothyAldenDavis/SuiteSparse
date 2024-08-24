#-------------------------------------------------------------------------------
# SuiteSparse/SPEX/CMakeLists.txt:  cmake for SPEX
#-------------------------------------------------------------------------------

# SPEX: (c) 2019-2024, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis, Texas A&M.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.22 )

set ( SPEX_DATE "Aug 20, 2024" )
set ( SPEX_VERSION_MAJOR 3 CACHE STRING "" FORCE )
set ( SPEX_VERSION_MINOR 2 CACHE STRING "" FORCE )
set ( SPEX_VERSION_SUB   1 CACHE STRING "" FORCE )

message ( STATUS "Building SPEX version: v"
    ${SPEX_VERSION_MAJOR}.
    ${SPEX_VERSION_MINOR}.
    ${SPEX_VERSION_SUB} " (" ${SPEX_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project ( SPEX
    VERSION "${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}"
    LANGUAGES C )

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find OpenMP
#-------------------------------------------------------------------------------

option ( SPEX_USE_OPENMP "ON: Use OpenMP in SPEX if available.  OFF: Do not use OpenMP.  (Default: SUITESPARSE_USE_OPENMP)" ${SUITESPARSE_USE_OPENMP} )
if ( SPEX_USE_OPENMP )
    if ( CMAKE_VERSION VERSION_LESS 3.24 )
        find_package ( OpenMP COMPONENTS C )
    else ( )
        find_package ( OpenMP COMPONENTS C GLOBAL )
    endif ( )
else ( )
    # OpenMP has been disabled
    set ( OpenMP_C_FOUND OFF )
endif ( )

if ( SPEX_USE_OPENMP AND OpenMP_C_FOUND )
    set ( SPEX_HAS_OPENMP ON )
else ( )
    set ( SPEX_HAS_OPENMP OFF )
endif ( )
message ( STATUS "SPEX has OpenMP: ${SPEX_HAS_OPENMP}" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND SPEX_USE_OPENMP AND NOT SPEX_HAS_OPENMP )
    message ( FATAL_ERROR "OpenMP required for SPEX but not found" )
endif ( )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.8.2
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.8.2 REQUIRED )
    endif ( )

    find_package ( AMD 3.3.3
        PATHS ${CMAKE_SOURCE_DIR}/../AMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::AMD )
        find_package ( AMD 3.3.3 REQUIRED )
    endif ( )

    find_package ( COLAMD 3.3.4
        PATHS ${CMAKE_SOURCE_DIR}/../COLAMD/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::COLAMD )
        find_package ( COLAMD 3.3.4 REQUIRED )
    endif ( )
endif ( )

find_package ( GMP 6.1.2 REQUIRED )     # from SPEX/cmake_modules
find_package ( MPFR 4.0.2 REQUIRED )    # from SPEX/cmake_modules

#-------------------------------------------------------------------------------
# configure files
#-------------------------------------------------------------------------------

configure_file ( "Config/SPEX.h.in"
    "${PROJECT_SOURCE_DIR}/Include/SPEX.h"
    NEWLINE_STYLE LF )
configure_file ( "Config/SPEX_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/SPEX_version.tex"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# include directories
#-------------------------------------------------------------------------------

include_directories ( Include
    SPEX_Backslash/Source SPEX_Cholesky/Source SPEX_LU/Source
    SPEX_Utilities/Source
    ${SUITESPARSE_CONFIG_INCLUDE_DIR}
    ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR}
    ${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} )

#-------------------------------------------------------------------------------
# dynamic spex library properties
#-------------------------------------------------------------------------------

file ( GLOB SPEX_SOURCES "SPEX*/Source/*.c" )

if ( BUILD_SHARED_LIBS )
    add_library ( SPEX SHARED ${SPEX_SOURCES} )

    set_target_properties ( SPEX PROPERTIES
        VERSION ${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME spex
        SOVERSION ${SPEX_VERSION_MAJOR}
        PUBLIC_HEADER "Include/SPEX.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( SPEX PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( SPEX 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# static spex library properties
#-------------------------------------------------------------------------------

if ( BUILD_STATIC_LIBS )
    add_library ( SPEX_static STATIC ${SPEX_SOURCES} )

    set_target_properties ( SPEX_static PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        OUTPUT_NAME spex
        PUBLIC_HEADER "Include/SPEX.h" )

    if ( MSVC OR ("${CMAKE_C_SIMULATE_ID}" STREQUAL "MSVC") )
        set_target_properties ( SPEX_static PROPERTIES
            OUTPUT_NAME spex_static )
    endif ( )

    if ( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.25" )
        set_target_properties ( SPEX_static PROPERTIES EXPORT_NO_SYSTEM ON )
    endif ( )

    target_include_directories ( SPEX_static 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )
endif ( )

#-------------------------------------------------------------------------------
# add the library dependencies
#-------------------------------------------------------------------------------

# SuiteSparseConfig:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( SPEX PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( SPEX_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

# AMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE SuiteSparse::AMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::AMD_static )
        target_link_libraries ( SPEX_static PRIVATE SuiteSparse::AMD_static )
    else ( )
        target_link_libraries ( SPEX_static PRIVATE SuiteSparse::AMD )
    endif ( )
endif ( )

# COLAMD:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE SuiteSparse::COLAMD )
endif ( )
if ( BUILD_STATIC_LIBS )
    if ( TARGET SuiteSparse::COLAMD_static )
        target_link_libraries ( SPEX_static PRIVATE SuiteSparse::COLAMD_static )
    else ( )
        target_link_libraries ( SPEX_static PRIVATE SuiteSparse::COLAMD )
    endif ( )
endif ( )

# MPFR:
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE ${MPFR_LIBRARIES} )
    target_include_directories ( SPEX SYSTEM AFTER PUBLIC ${MPFR_INCLUDE_DIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND SPEX_STATIC_LIBS ${MPFR_STATIC} )
    target_link_libraries ( SPEX_static PUBLIC ${MPFR_STATIC} )
    target_include_directories ( SPEX_static SYSTEM AFTER PUBLIC ${MPFR_INCLUDE_DIR} )
endif ( )

# GMP:
# must occur after MPFR
if ( BUILD_SHARED_LIBS )
    target_link_libraries ( SPEX PRIVATE ${GMP_LIBRARIES} )
    target_include_directories ( SPEX SYSTEM AFTER PUBLIC ${GMP_INCLUDE_DIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    list ( APPEND SPEX_STATIC_LIBS ${GMP_STATIC} )
    target_link_libraries ( SPEX_static PUBLIC ${GMP_STATIC} )
    target_include_directories ( SPEX_static SYSTEM AFTER PUBLIC ${GMP_INCLUDE_DIR} )
endif ( )

# OpenMP:
if ( SPEX_HAS_OPENMP )
    message ( STATUS "OpenMP C libraries:      ${OpenMP_C_LIBRARIES}" )
    message ( STATUS "OpenMP C include:        ${OpenMP_C_INCLUDE_DIRS}" )
    message ( STATUS "OpenMP C flags:          ${OpenMP_C_FLAGS}" )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( SPEX PRIVATE OpenMP::OpenMP_C )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( SPEX_static PRIVATE OpenMP::OpenMP_C )
        list ( APPEND SPEX_STATIC_LIBS ${OpenMP_C_LIBRARIES} )
    endif ( )
else ( )
    # use threadprivate variables for SPEX instead of globals, so multiple user
    # threads can call SPEX in parallel on different matrices.  Otherwise, SPEX
    # is not thread-safe (see SPEX_Utilities/Source/SPEX_gmp.c).
    include ( SuiteSparse__thread )
endif ( )

# libm:
include ( CheckSymbolExists )
check_symbol_exists ( fmax "math.h" NO_LIBM )
if ( NOT NO_LIBM )
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( SPEX PRIVATE m )
    endif ( )
    if ( BUILD_STATIC_LIBS )
        list ( APPEND SPEX_STATIC_LIBS "m" )
        target_link_libraries ( SPEX_static PUBLIC m )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# SPEX installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS SPEX
        EXPORT SPEXTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS SPEX_static
        EXPORT SPEXTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )

# create (temporary) export target file during build
export ( EXPORT SPEXTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/SPEXTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT SPEXTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SPEX )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/SPEXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/SPEXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/SPEXConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SPEX )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/SPEXConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SPEXConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindGMP.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/FindMPFR.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/SPEX )

#-------------------------------------------------------------------------------
# configure MATLAB
#-------------------------------------------------------------------------------

cmake_path ( IS_ABSOLUTE SUITESPARSE_LIBDIR SUITESPARSE_LIBDIR_IS_ABSOLUTE )
if (SUITESPARSE_LIBDIR_IS_ABSOLUTE)
    set ( matlab_libdir "${SUITESPARSE_LIBDIR}")
else ( )
    set ( matlab_libdir "${CMAKE_INSTALL_PREFIX}/${SUITESPARSE_LIBDIR}")
endif ( )
cmake_path ( IS_ABSOLUTE SUITESPARSE_INCLUDEDIR SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE )
if (SUITESPARSE_INCLUDEDIR_IS_ABSOLUTE)
    set ( matlab_includedir "${SUITESPARSE_INCLUDEDIR}")
else ( )
    set ( matlab_includedir "${CMAKE_INSTALL_PREFIX}/${SUITESPARSE_INCLUDEDIR}")
endif ( )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
    # This might be something like:
    #   /usr/lib/libgomp.so;/usr/lib/libpthread.a;m
    # convert to -l flags for pkg-config, i.e.: "-lgomp -lpthread -lm"
    set ( SPEX_STATIC_LIBS_LIST ${SPEX_STATIC_LIBS} )
    set ( SPEX_STATIC_LIBS "" )
    foreach ( _lib ${SPEX_STATIC_LIBS_LIST} )
        string ( FIND ${_lib} "." _pos REVERSE )
        if ( ${_pos} EQUAL "-1" )
            set ( SPEX_STATIC_LIBS "${SPEX_STATIC_LIBS} -l${_lib}" )
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
                    set ( SPEX_STATIC_LIBS "${SPEX_STATIC_LIBS} -l${_libname}" )
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
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:SPEX> )
    else ( )
        set ( SUITESPARSE_LIB_BASE_NAME $<TARGET_FILE_BASE_NAME:SPEX_static> )
    endif ( )
    configure_file (
        Config/SPEX.pc.in
        SPEX.pc.out
        @ONLY
        NEWLINE_STYLE LF )
    file ( GENERATE
        OUTPUT SPEX.pc
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/SPEX.pc.out
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SPEX.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# Python interface
#-------------------------------------------------------------------------------

# The Python interface is built by default.  It can be disabled with
# SPEX_USE_PYTHON (for just SPEX) or SUITESPARSE_USE_PYTHON (for all of
# SuiteSparse).  Currently, only SPEX has a Python interface in SuiteSparse.

if ( BUILD_SHARED_LIBS )
    set ( _spex_use_python_default ${SUITESPARSE_USE_PYTHON} )
else ( )
    set ( _spex_use_python_default OFF )
endif ( )

option ( SPEX_USE_PYTHON "ON (default if building shared libraries): build Python interface for SPEX.  OFF (default if not building shared libraries): do not build Python interface for SPEX" ${_spex_use_python_default} )

if ( SPEX_USE_PYTHON AND BUILD_SHARED_LIBS )
    set ( SPEX_HAS_PYTHON ON )
else ( )
    set ( SPEX_HAS_PYTHON OFF )
endif ( )
message ( STATUS "SPEX with Python interface: ${SPEX_HAS_PYTHON}" )

# check for strict usage
if ( SUITESPARSE_USE_STRICT AND SPEX_USE_PYTHON AND NOT SPEX_HAS_PYTHON )
    message ( FATAL_ERROR "Python interface for SPEX requires building shared libraries" )
endif ( )

if ( SPEX_HAS_PYTHON )

    file ( GLOB SPEX_PYTHON_SOURCES "Python/SPEXpy/Source/*.c" )
    add_library ( spexpython SHARED ${SPEX_PYTHON_SOURCES} )

    set_target_properties ( spexpython PROPERTIES
        VERSION ${SPEX_VERSION_MAJOR}.${SPEX_VERSION_MINOR}.${SPEX_VERSION_SUB}
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        SOVERSION ${SPEX_VERSION_MAJOR}
        PUBLIC_HEADER "Python/SPEXpy/Source/spex_python_connect.h"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    # MPFR:
    target_link_libraries ( spexpython PRIVATE ${MPFR_LIBRARIES} )

    # GMP:
    # must occur after MPFR
    target_link_libraries ( spexpython PRIVATE ${GMP_LIBRARIES} )

    target_link_libraries ( spexpython PRIVATE SuiteSparse::COLAMD )
    target_link_libraries ( spexpython PRIVATE SuiteSparse::AMD )
    target_link_libraries ( spexpython PRIVATE SPEX )

    install ( TARGETS spexpython
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

endif ( )

#-------------------------------------------------------------------------------
# Demo library and programs
#-------------------------------------------------------------------------------

if ( SUITESPARSE_DEMOS )

    #---------------------------------------------------------------------------
    # demo library
    #---------------------------------------------------------------------------

    message ( STATUS "Also compiling the demos in SPEX/Demo" )

    #---------------------------------------------------------------------------
    # Demo programs
    #---------------------------------------------------------------------------

    include_directories ( Demo )
    file ( GLOB SPEX_DEMO_SOURCES "Demo/Utilities/*.c" )

    add_executable ( spex_demo_backslash         "Demo/spex_demo_backslash.c"         ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_cholesky_extended "Demo/spex_demo_cholesky_extended.c" ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_cholesky_simple   "Demo/spex_demo_cholesky_simple.c"   ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_ldl_extended      "Demo/spex_demo_ldl_extended.c"   ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_ldl_simple        "Demo/spex_demo_ldl_simple.c"   ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_lu_doub           "Demo/spex_demo_lu_doub.c"           ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_lu_extended       "Demo/spex_demo_lu_extended.c"       ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_lu_simple1        "Demo/spex_demo_lu_simple1.c"        ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_lu_simple2        "Demo/spex_demo_lu_simple2.c"        ${SPEX_DEMO_SOURCES} )
    add_executable ( spex_demo_threaded          "Demo/spex_demo_threaded.c"          ${SPEX_DEMO_SOURCES} )

    # Libraries required for Demo programs
    if ( BUILD_SHARED_LIBS )
        target_link_libraries ( spex_demo_backslash         PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_cholesky_extended PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_cholesky_simple   PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_ldl_extended      PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_ldl_simple        PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_lu_doub           PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_lu_extended       PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_lu_simple1        PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_lu_simple2        PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
        target_link_libraries ( spex_demo_threaded          PUBLIC SPEX SuiteSparse::SuiteSparseConfig ${MPFR_LIBRARIES} ${GMP_LIBRARIES} SuiteSparse::AMD SuiteSparse::COLAMD )
    else ( )
        target_link_libraries ( spex_demo_backslash         PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_cholesky_extended PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_cholesky_simple   PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_ldl_simple        PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_ldl_extended      PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_lu_doub           PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_lu_extended       PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_lu_simple1        PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_lu_simple2        PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
        target_link_libraries ( spex_demo_threaded          PUBLIC SPEX_static SuiteSparse::AMD_static SuiteSparse::COLAMD_static )
    endif ( )

    if ( SPEX_HAS_OPENMP )
        if ( BUILD_SHARED_LIBS )
            target_link_libraries ( spex_demo_threaded PUBLIC OpenMP::OpenMP_C )
        endif ( )
        if ( BUILD_STATIC_LIBS )
            target_link_libraries ( spex_demo_threaded PUBLIC OpenMP::OpenMP_C )
        endif ( )
    endif ( )

else ( )

    message ( STATUS "Skipping the demos in SPEX/Demo" )

endif ( )

#-------------------------------------------------------------------------------
# configure MATLAB and Tcov
#-------------------------------------------------------------------------------

if ( BUILD_SHARED_LIBS AND NOT SUITESPARSE_ROOT_CMAKELISTS )

    # get paths to libraries
    get_filename_component ( GMP_PATH ${GMP_LIBRARY} DIRECTORY )
    get_filename_component ( SUITESPARSE_CONFIG_PATH ${SUITESPARSE_CONFIG_LIBRARY} DIRECTORY )
    get_filename_component ( COLAMD_PATH ${COLAMD_LIBRARY} DIRECTORY )
    get_filename_component ( AMD_PATH ${AMD_LIBRARY} DIRECTORY )
    get_filename_component ( MPFR_PATH ${MPFR_LIBRARY} DIRECTORY )

    # message ( STATUS "suitesparseconfig path ${SUITESPARSE_CONFIG_PATH}" )
    # message ( STATUS "colamd path ${COLAMD_PATH}" )
    # message ( STATUS "amd path ${AMD_PATH}" )
    # message ( STATUS "gmp path ${GMP_PATH}" )
    # message ( STATUS "mpfr path ${MPFR_PATH}" )

    # construct the -I list
    get_target_property ( INCS SPEX INCLUDE_DIRECTORIES )
    list ( TRANSFORM INCS PREPEND " -I" )
    set ( SPEX_INCS "" )
    foreach ( INC ${INCS} )
    #   message ( STATUS "check inc: ${INC}" )
        if ( NOT ${INC} MATCHES "TARGET_PROPERTY" )
    #       message ( STATUS "add   inc: ${INC}" )
            string ( APPEND SPEX_INCS " " ${INC} )
        endif ( )
    endforeach ( )
    # message ( STATUS "Incs: ${SPEX_INCS}" )

    # construct the library list for Tcov/Makefile
    set ( SPEX_LIB_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}" )
    get_target_property ( LIBS SPEX LINK_LIBRARIES )
    string ( REPLACE "." "\\." LIBSUFFIX ${SPEX_LIB_SUFFIX} )
    set ( SPEX_LIBS "" )
    foreach ( LIB_NAME ${LIBS} )
    #   message ( STATUS "Lib: ${LIB_NAME}" )
        if ( LIB_NAME MATCHES "::" )
            # do nothing
        elseif ( LIB_NAME MATCHES ${LIBSUFFIX} )
            # message ( STATUS "has suffix" )
            string ( APPEND SPEX_LIBS " " ${LIB_NAME} )
        else ( )
            # message ( STATUS "no suffix" )
            string ( APPEND SPEX_LIBS " -l" ${LIB_NAME} )
        endif ( )
    endforeach ( )
    # message ( STATUS "Libs: ${SPEX_LIBS}" )

    configure_file (
            "Config/spex_deps.m.in"
            "${PROJECT_SOURCE_DIR}/MATLAB/spex_deps.m"
            @ONLY
            NEWLINE_STYLE LF )

    configure_file (
            "Config/Tcov_Makefile.in"
            "${PROJECT_SOURCE_DIR}/Tcov/Makefile"
            @ONLY
            NEWLINE_STYLE LF )

endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

include ( SuiteSparseReport )

