#-------------------------------------------------------------------------------
# Mongoose/CMakeLists.txt: instructions for building Mongoose with cmake
#-------------------------------------------------------------------------------
#
# Mongoose, Copyright (c) 2018-2023, All Rights Reserved.
#   Nuri Yeralan, Microsoft Research
#   Scott Kolodziej, Texas A&M University
#   Tim Davis, Texas A&M University
#   William Hager, University of Florida.
#
# SPDX-License-Identifier: GPL-3.0-only
#
# http://suitesparse.com   See Mongoose/Doc/License.txt for license.
#
# To compile the Mongoose library and demo programs just do "make"
# in this directory.  The Mongoose/Makefile will then run cmake
# in the build directory.
#
# To install Mongoose in /usr/local/lib and /usr/local/include,
# and also in ../lib, ../include, and ../bin:
#
#   sudo make install
#
# To install only in ../lib and ../include (no sudo required):
#
#   make local ; make install
#
# To remove all compiled files and libraries (except installed ones):
#
#   make distclean

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 )

set ( Mongoose_DATE "Dec 30, 2023" )
set ( Mongoose_NUMERIC_DATE "2023-12-30" )
set ( Mongoose_VERSION_MAJOR 3 CACHE STRING "" FORCE )
set ( Mongoose_VERSION_MINOR 3 CACHE STRING "" FORCE )
set ( Mongoose_VERSION_PATCH 0 CACHE STRING "" FORCE )

message ( STATUS "Building Mongoose version: v"
    ${Mongoose_VERSION_MAJOR}.
    ${Mongoose_VERSION_MINOR}.
    ${Mongoose_VERSION_PATCH} " (" ${Mongoose_DATE} ")" )

#-------------------------------------------------------------------------------
# define the project
#-------------------------------------------------------------------------------

project(Mongoose
        VERSION "${Mongoose_VERSION_MAJOR}.${Mongoose_VERSION_MINOR}.${Mongoose_VERSION_PATCH}"
        LANGUAGES CXX C)

#-------------------------------------------------------------------------------
# SuiteSparse policies
#-------------------------------------------------------------------------------

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${PROJECT_SOURCE_DIR}/../SuiteSparse_config/cmake_modules )

include ( SuiteSparsePolicy )

#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    find_package ( SuiteSparse_config 7.4.0
        PATHS ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/build NO_DEFAULT_PATH )
    if ( NOT TARGET SuiteSparse::SuiteSparseConfig )
        find_package ( SuiteSparse_config 7.4.0 REQUIRED )
    endif ( )
endif ( )

# For ctests
find_package ( Python COMPONENTS Interpreter )

#-------------------------------------------------------------------------------

# Mongoose installation location
include ( GNUInstallDirs )

# Configure files to pass CMake information
configure_file (
        "Version/Mongoose.hpp.in"
        "${PROJECT_SOURCE_DIR}/Include/Mongoose.hpp"
        NEWLINE_STYLE LF
)
configure_file (
        "Version/title-info.tex.in"
        "${PROJECT_SOURCE_DIR}/Doc/title-info.tex"
        NEWLINE_STYLE LF
)
configure_file (
        "Version/codemeta.json.in"
        "${PROJECT_SOURCE_DIR}/codemeta.json"
        NEWLINE_STYLE LF
)

include_directories("${PROJECT_BINARY_DIR}")

set(MONGOOSE_FILES
        Include/Mongoose_BoundaryHeap.hpp
        Include/Mongoose_Coarsening.hpp
        Include/Mongoose_CSparse.hpp
        Include/Mongoose_CutCost.hpp
        Include/Mongoose_Debug.hpp
        Include/Mongoose_EdgeCutOptions.hpp
        Include/Mongoose_EdgeCutProblem.hpp
        Include/Mongoose_EdgeCut.hpp
        Include/Mongoose_Graph.hpp
        Include/Mongoose_GuessCut.hpp
        Include/Mongoose_ImproveFM.hpp
        Include/Mongoose_ImproveQP.hpp
        Include/Mongoose_Internal.hpp
        Include/Mongoose_IO.hpp
        Include/Mongoose_Logger.hpp
        Include/Mongoose_Matching.hpp
        Include/Mongoose_Random.hpp
        Include/Mongoose_Refinement.hpp
        Include/Mongoose_Sanitize.hpp
        Include/Mongoose_Waterdance.hpp
        Source/Mongoose_BoundaryHeap.cpp
        Source/Mongoose_Coarsening.cpp
        Source/Mongoose_CSparse.cpp
        Source/Mongoose_Debug.cpp
        Source/Mongoose_EdgeCut.cpp
        Source/Mongoose_Graph.cpp
        Source/Mongoose_GuessCut.cpp
        Source/Mongoose_ImproveFM.cpp
        Source/Mongoose_ImproveQP.cpp
        Source/Mongoose_IO.cpp
        Source/Mongoose_Logger.cpp
        Source/Mongoose_Matching.cpp
        Source/Mongoose_EdgeCutOptions.cpp
        Source/Mongoose_EdgeCutProblem.cpp
        Source/Mongoose_EdgeCut.cpp
        Source/Mongoose_Random.cpp
        Source/Mongoose_Refinement.cpp
        Source/Mongoose_Sanitize.cpp
        Source/Mongoose_Version.cpp
        Source/Mongoose_Waterdance.cpp
        )

set(NAPHEAP_FILES
        Include/Mongoose_QPBoundary.hpp
        Include/Mongoose_QPDelta.hpp
        Include/Mongoose_QPGradProj.hpp
        Include/Mongoose_QPLinks.hpp
        Include/Mongoose_QPMaxHeap.hpp
        Include/Mongoose_QPMinHeap.hpp
        Include/Mongoose_QPNapDown.hpp
        Include/Mongoose_QPNapsack.hpp
        Include/Mongoose_QPNapUp.hpp
        Source/Mongoose_QPBoundary.cpp
        Source/Mongoose_QPDelta.cpp
        Source/Mongoose_QPGradProj.cpp
        Source/Mongoose_QPLinks.cpp
        Source/Mongoose_QPMaxHeap.cpp
        Source/Mongoose_QPMinHeap.cpp
        Source/Mongoose_QPNapDown.cpp
        Source/Mongoose_QPNapsack.cpp
        Source/Mongoose_QPNapUp.cpp
        )

set(MMIO_FILES
        External/mmio/Source/mmio.c
        External/mmio/Include/mmio.h)

set(MONGOOSE_LIB_FILES
        ${MONGOOSE_FILES}
        ${NAPHEAP_FILES}
        ${MMIO_FILES})

set(EXE_FILES
        Executable/mongoose.cpp)

set(DEMO_FILES
        Demo/demo.cpp
        Include/Mongoose.hpp)

# Specify the Include directory for *.hpp files
include_directories(Include
        External/mmio/Include
        )

# Require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set the output directories
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if ( BUILD_STATIC_LIBS )
    # Build the Mongoose library
    add_library ( Mongoose_static STATIC ${MONGOOSE_LIB_FILES} )
#   set_property ( TARGET Mongoose_static PROPERTY POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( Mongoose_static PROPERTIES
        OUTPUT_NAME mongoose
        PUBLIC_HEADER "Include/Mongoose.hpp" )

    if ( MSVC )
        set_target_properties ( Mongoose_static PROPERTIES
            OUTPUT_NAME mongoose_static )
    endif ( )

    target_include_directories ( Mongoose_static
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( Mongoose_static PUBLIC SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( Mongoose_static PUBLIC SuiteSparse::SuiteSparseConfig )
    endif ( )
endif ( )

if ( BUILD_SHARED_LIBS )
    # Build the Mongoose library for dynamic linking
    add_library ( Mongoose SHARED ${MONGOOSE_LIB_FILES} )
    set_property ( TARGET Mongoose PROPERTY POSITION_INDEPENDENT_CODE ON )
    set_target_properties ( Mongoose PROPERTIES
        OUTPUT_NAME mongoose
        SOVERSION ${Mongoose_VERSION_MAJOR}
        PUBLIC_HEADER "Include/Mongoose.hpp"
        WINDOWS_EXPORT_ALL_SYMBOLS ON )

    target_include_directories ( Mongoose
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
                  $<INSTALL_INTERFACE:${SUITESPARSE_INCLUDEDIR}> )

    target_link_libraries ( Mongoose PRIVATE SuiteSparse::SuiteSparseConfig )
    target_include_directories ( Mongoose PUBLIC
        "$<TARGET_PROPERTY:SuiteSparse::SuiteSparseConfig,INTERFACE_INCLUDE_DIRECTORIES>" )

    target_include_directories ( Mongoose PRIVATE . )
endif ( )

#-------------------------------------------------------------------------------

# Build the Mongoose debug/test library
add_library ( Mongoose_static_dbg STATIC ${MONGOOSE_LIB_FILES} )
set_target_properties ( Mongoose_static_dbg PROPERTIES
        OUTPUT_NAME mongoose_dbg )

if ( TARGET SuiteSparse::SuiteSparseConfig_static )
    target_link_libraries ( Mongoose_static_dbg SuiteSparse::SuiteSparseConfig_static )
else ( )
    target_link_libraries ( Mongoose_static_dbg SuiteSparse::SuiteSparseConfig )
endif ( )

# Build the Mongoose executable
add_executable ( mongoose_exe ${EXE_FILES} )
set_target_properties ( mongoose_exe PROPERTIES
        OUTPUT_NAME mongoose )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( mongoose_exe Mongoose_static )
else ( )
    target_link_libraries ( mongoose_exe Mongoose )
endif ( )
if ( TARGET SuiteSparse::SuiteSparseConfig_static )
    target_link_libraries ( mongoose_exe SuiteSparse::SuiteSparseConfig_static )
else ( )
    target_link_libraries ( mongoose_exe SuiteSparse::SuiteSparseConfig )
endif ( )

# Build the Demo executable
add_executable ( demo_exe ${DEMO_FILES} )
set_target_properties ( demo_exe PROPERTIES
        OUTPUT_NAME demo )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( demo_exe Mongoose_static )
else ( )
    target_link_libraries ( demo_exe Mongoose )
endif ( )
if ( TARGET SuiteSparse::SuiteSparseConfig_static )
    target_link_libraries ( demo_exe SuiteSparse::SuiteSparseConfig_static )
else ( )
    target_link_libraries ( demo_exe SuiteSparse::SuiteSparseConfig )
endif ( )

# Coverage and Unit Testing Setup
enable_testing ( )
set ( TESTING_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests )

if ( Python_Interpreter_FOUND )
    # I/O Tests
    add_executable ( mongoose_test_io
        Tests/Mongoose_Test_IO.cpp
        Tests/Mongoose_Test_IO_exe.cpp )
    target_link_libraries ( mongoose_test_io Mongoose_static_dbg )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( mongoose_test_io SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( mongoose_test_io SuiteSparse::SuiteSparseConfig )
    endif ( )
    set_target_properties ( mongoose_test_io PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )

    add_test ( NAME Mongoose_IO_Test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -min 1 -max 15 -t io -k )

    if ( WIN32 AND BUILD_SHARED_LIBS )
        set_tests_properties ( Mongoose_IO_Test PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Mongoose>" )
    endif ( )

    # Edge Separator Tests
    add_executable ( mongoose_test_edgesep
        Tests/Mongoose_Test_EdgeSeparator.cpp
        Tests/Mongoose_Test_EdgeSeparator_exe.cpp)
    target_link_libraries ( mongoose_test_edgesep Mongoose_static_dbg )
    set_target_properties ( mongoose_test_edgesep PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )

    add_test ( NAME Mongoose_Edge_Separator_Test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -min 1 -max 15 -t edgesep )
    add_test ( NAME Mongoose_Edge_Separator_Test_2
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -t edgesep -i 21 39 191 1557 1562 353 2468 1470 1380 505 182 201 2331 760 1389 2401 2420 242 250 1530 1533 360 1437 )
    add_test ( NAME Mongoose_Weighted_Edge_Separator_Test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -t edgesep -i 2624 )
    add_test ( NAME Mongoose_Target_Split_Test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -min 1 -max 15 -t edgesep -s 0.3 )

    if ( WIN32 AND BUILD_SHARED_LIBS )
        set_tests_properties ( Mongoose_Edge_Separator_Test Mongoose_Edge_Separator_Test_2 Mongoose_Weighted_Edge_Separator_Test Mongoose_Target_Split_Test PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Mongoose>" )
    endif ( )

    # Memory Tests
    add_executable ( mongoose_test_memory
        Tests/Mongoose_Test_Memory.cpp
        Tests/Mongoose_Test_Memory_exe.cpp)
    target_link_libraries ( mongoose_test_memory Mongoose_static_dbg )
    set_target_properties ( mongoose_test_memory PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )

    add_test ( NAME Mongoose_Memory_Test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -min 1 -max 15 -t memory )

    if ( WIN32 AND BUILD_SHARED_LIBS )
        set_tests_properties ( Mongoose_Memory_Test PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Mongoose>" )
    endif ( )

    # Performance Test
    add_executable ( mongoose_test_performance
        Tests/Mongoose_Test_Performance.cpp
        Tests/Mongoose_Test_Performance_exe.cpp )
    if ( BUILD_STATIC_LIBS )
        target_link_libraries ( mongoose_test_performance Mongoose_static )
    else ( )
        target_link_libraries ( mongoose_test_performance Mongoose )
    endif ( )
    if ( TARGET SuiteSparse::SuiteSparseConfig_static )
        target_link_libraries ( mongoose_test_performance SuiteSparse::SuiteSparseConfig_static )
    else ( )
        target_link_libraries ( mongoose_test_performance SuiteSparse::SuiteSparseConfig )
    endif ( )
    set_target_properties ( mongoose_test_performance PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )

    add_test ( NAME Mongoose_Performance_Test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -min 1 -max 15 -t performance -p )
    add_test ( NAME Mongoose_Performance_Test_2
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/runTests -t performance -i 21 39 1557 1562 353 2468 1470 1380 505 182 201 2331 760 1389 2401 2420 242 250 1530 1533 -p )

    if ( WIN32 AND BUILD_SHARED_LIBS )
        set_tests_properties ( Mongoose_Performance_Test Mongoose_Performance_Test_2 PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Mongoose>" )
    endif ( )
endif ( )

# Reference Test
add_executable ( mongoose_test_reference
    Tests/Mongoose_Test_Reference.cpp
    Tests/Mongoose_Test_Reference_exe.cpp )
if ( BUILD_STATIC_LIBS )
    target_link_libraries ( mongoose_test_reference Mongoose_static )
else ( )
    target_link_libraries ( mongoose_test_reference Mongoose )
endif ( )
if ( TARGET SuiteSparse::SuiteSparseConfig_static )
    target_link_libraries ( mongoose_test_reference SuiteSparse::SuiteSparseConfig_static )
else ( )
    target_link_libraries ( mongoose_test_reference SuiteSparse::SuiteSparseConfig )
endif ( )
set_target_properties ( mongoose_test_reference PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )

# Unit Tests
add_executable ( mongoose_unit_test_io
    Tests/Mongoose_UnitTest_IO_exe.cpp )
target_link_libraries ( mongoose_unit_test_io Mongoose_static_dbg )
set_target_properties ( mongoose_unit_test_io PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )
add_test ( NAME Mongoose_Unit_Test_IO
    COMMAND ${TESTING_OUTPUT_PATH}/mongoose_unit_test_io
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Tests )

add_executable ( mongoose_unit_test_graph
    Tests/Mongoose_UnitTest_Graph_exe.cpp )
target_link_libraries ( mongoose_unit_test_graph Mongoose_static_dbg )
set_target_properties ( mongoose_unit_test_graph PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )
add_test ( NAME Mongoose_Unit_Test_Graph
    COMMAND ${TESTING_OUTPUT_PATH}/mongoose_unit_test_graph
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Tests )

add_executable ( mongoose_unit_test_edgesep
    Tests/Mongoose_UnitTest_EdgeSep_exe.cpp )
target_link_libraries ( mongoose_unit_test_edgesep Mongoose_static_dbg )
set_target_properties ( mongoose_unit_test_edgesep PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTING_OUTPUT_PATH} )
add_test ( NAME Mongoose_Unit_Test_EdgeSep
    COMMAND ${TESTING_OUTPUT_PATH}/mongoose_unit_test_edgesep
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Tests )

if ( WIN32 AND BUILD_SHARED_LIBS )
    set_tests_properties ( Mongoose_Unit_Test_IO Mongoose_Unit_Test_Graph Mongoose_Unit_Test_EdgeSep PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:Mongoose>" )
endif ( )

if ( $ENV{MONGOOSE_COVERAGE} )
    set ( COV ON )
else ( )
    set ( COV OFF )
endif ( )

option ( MONGOOSE_COVERAGE "OFF: do not compile debug library with test coverage.  ON: debug with test coverage" ${COV} )
message ( STATUS "test coverage for debug library: ${MONGOOSE_COVERAGE}" )

message(STATUS "CMAKE_CXX_COMPILER: " ${BoldBlue} ${CMAKE_CXX_COMPILER_ID} ${ColourReset})
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    # using Clang
#   SET(CMAKE_CXX_FLAGS "-O3 -fwrapv")
    # Debug flags for Clang
#   SET(CMAKE_CXX_FLAGS_DEBUG "-g -fwrapv")
#   SET(CMAKE_C_FLAGS_DEBUG "-g")
#   SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g")
    if ( MONGOOSE_COVERAGE )
        SET ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage" ) 
        SET ( CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} --coverage" )
        SET ( CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage" )
    endif ( )
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
#   SET(CMAKE_CXX_FLAGS "-O3 -fwrapv")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-unused-variable" )
    # Debug flags for GCC
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.6")
#       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        message(WARNING "${BoldRed} WARNING:${ColourReset} Your compiler does not support the C++11 ISO standard. Future versions of Mongoose will require a compiler with C++11 support. We recommend you upgrade to at least GCC 4.6.")
    endif ()
#   SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -fwrapv -W -Wshadow -Wunused-parameter -Wunused-function -Wunused -Wno-system-headers -Wno-deprecated -Woverloaded-virtual -Wwrite-strings ")
#   SET(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -fwrapv -W ")
#   SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -W ")
#   SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "")
    if ( MONGOOSE_COVERAGE )
        SET (CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG} -Wall -W -Wshadow -Wunused-parameter -Wunused-function -Wunused -Wno-system-headers -Wno-deprecated -Woverloaded-virtual -Wwrite-strings ")
        SET ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage" ) 
        SET ( CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage" )
        SET ( CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage" )
    endif ( )
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
#   SET(CMAKE_CXX_FLAGS "-O3 -no-prec-div -xHOST -ipo -fwrapv")
    # Debug flags for Intel
#   SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -fwrapv")
#   SET(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall")
#elseif ( MSVC )
    # using Visual Studio C++
endif ()

set_target_properties ( Mongoose_static_dbg PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
    LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )

# Add debug compile/linker flags
if ( Python_Interpreter_FOUND )
    set_target_properties ( mongoose_test_io mongoose_test_memory mongoose_test_edgesep PROPERTIES
        COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
        LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )
endif ( )
set_target_properties ( mongoose_unit_test_io mongoose_unit_test_graph mongoose_unit_test_edgesep PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
    LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )

set ( CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1 ) # Necessary for gcov - prevents file.cpp.gcda instead of file.gcda

# if ( BUILD_STATIC_LIBS )
#   add_custom_command ( TARGET Mongoose_static
#           POST_BUILD
#           COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Mongoose_static> ${PROJECT_SOURCE_DIR}/Lib
#           COMMENT "Copying libmongoose (static) to root Lib directory"
#           )
# endif ( )

# add_custom_command ( TARGET Mongoose
#       POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Mongoose> ${PROJECT_SOURCE_DIR}/Lib
#       COMMENT "Copying libmongoose (dynamic) to root Lib directory"
#       )

add_custom_target ( purge
        COMMAND rm -rf ${PROJECT_BINARY_DIR}/*
#       COMMAND rm -f ${PROJECT_SOURCE_DIR}/Lib/libmongoose.*
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/title-info.tex
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/title-info.tex.aux
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.idx
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.log
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.out
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.aux
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.toc
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.bbl
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Doc/Mongoose_UserGuide.blg
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Matrix/*.tar.gz
        COMMAND rm -f ${PROJECT_SOURCE_DIR}/Matrix/*.csv
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        )

add_custom_target(userguide
        COMMAND make
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Doc
        )

#-------------------------------------------------------------------------------
# Mongoose installation location
#-------------------------------------------------------------------------------

include ( CMakePackageConfigHelpers )

if ( BUILD_SHARED_LIBS )
    install ( TARGETS Mongoose
        EXPORT MongooseTargets
        LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
if ( BUILD_STATIC_LIBS )
    install ( TARGETS Mongoose_static
        EXPORT MongooseTargets
        ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
        PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )
endif ( )
install ( TARGETS mongoose_exe
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR} )

# create (temporary) export target file during build
export ( EXPORT MongooseTargets
    NAMESPACE SuiteSparse::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/MongooseTargets.cmake )

# install export target, config and version files for find_package
install ( EXPORT MongooseTargets
    NAMESPACE SuiteSparse::
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/Mongoose )

# generate config file to be used in common build tree
set ( SUITESPARSE_IN_BUILD_TREE ON )
configure_package_config_file (
    Config/MongooseConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MongooseConfig.cmake
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/MongooseConfig.cmake )

# generate config file to be installed
set ( SUITESPARSE_IN_BUILD_TREE OFF )
configure_package_config_file (
    Config/MongooseConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/target/MongooseConfig.cmake
    INSTALL_DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/Mongoose )

write_basic_package_version_file (
    ${CMAKE_CURRENT_BINARY_DIR}/MongooseConfigVersion.cmake
    COMPATIBILITY SameMajorVersion )

install ( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/target/MongooseConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MongooseConfigVersion.cmake
    DESTINATION ${SUITESPARSE_PKGFILEDIR}/cmake/Mongoose )

#-------------------------------------------------------------------------------
# create pkg-config file
#-------------------------------------------------------------------------------

if ( NOT MSVC )
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
        Config/Mongoose.pc.in
        Mongoose.pc
        @ONLY
        NEWLINE_STYLE LF )
    install ( FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Mongoose.pc
        DESTINATION ${SUITESPARSE_PKGFILEDIR}/pkgconfig )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    include ( SuiteSparseReport )
endif ( )
