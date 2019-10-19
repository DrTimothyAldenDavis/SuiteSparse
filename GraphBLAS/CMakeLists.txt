# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
# http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.
#
# CMakeLists.txt: instructions for cmake to build GraphBLAS.
# An ANSI C11 compiler is required.
#
# To compile the GraphBLAS library and demo programs:
#
#   cd build
#   cmake ..
#   make
#
# If that fails for any reason, make sure your compiler support ANSI C11.  You
# could try changing your compiler, for example, use one of these commands:
#
#   CC=icc cmake ..
#   CC=xlc cmake ..
#   CC=gcc cmake ..
#
# To run the demos:
#
#   cd ../Demo
#   ./demo
#
# To install GraphBLAS in /usr/local/lib and /usr/local/include:
#
#   cd build
#   sudo make install
#
# To remove all compiled files and libraries (except installed ones):
#
#   cd build
#   rm -rf *

# cmake 3.0 is preferred.
cmake_minimum_required ( VERSION 2.8.12 )

if ( CMAKE_VERSION VERSION_GREATER "3.0" )
    cmake_policy ( SET CMP0042 NEW )
    cmake_policy ( SET CMP0048 NEW )
endif ( )

# version of SuiteSparse:GraphBLAS
set ( GraphBLAS_DATE "July 4, 2018" )
set ( GraphBLAS_VERSION_MAJOR 2 )
set ( GraphBLAS_VERSION_MINOR 0 )
set ( GraphBLAS_VERSION_SUB   3 )

# GraphBLAS C API Specification version, at graphblas.org
set ( GraphBLAS_API_DATE "Nov 14, 2017" )
set ( GraphBLAS_API_VERSION_MAJOR 1 )
set ( GraphBLAS_API_VERSION_MINOR 1 )
set ( GraphBLAS_API_VERSION_SUB   0 )

if ( CMAKE_MAJOR_VERSION GREATER 2 )
    project ( graphblas
        VERSION "${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}" )
else ( )
    project ( graphblas )
endif ( )

# Configure a header file to pass CMake information
configure_file (
    "Version/GraphBLAS.h.in" 
    "${PROJECT_SOURCE_DIR}/Include/GraphBLAS.h"
)
configure_file (
    "Version/GraphBLAS_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/GraphBLAS_version.tex"
)
configure_file (
    "Version/GraphBLAS_API_version.tex.in"
    "${PROJECT_SOURCE_DIR}/Doc/GraphBLAS_API_version.tex"
)
configure_file (
    "Version/README.txt.in"
    "${PROJECT_SOURCE_DIR}/README.txt"
)

include ( GNUInstallDirs )

if (NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

# For development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

# include directories for both graphblas and graphblasdemo libraries
include_directories ( Source/Template Source Source/Generated Include Demo/Include )

# check which compiler is being used.  If you need to make
# compiler-specific modifications, here is the place to do it.
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # cmake 2.8 workaround: gcc needs to be told to do ANSI C11.
    # cmake 3.0 doesn't have this problem.
    set (CMAKE_C_FLAGS  "-std=c11 -lm -fopenmp")
    if (CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9)
        message (FATAL_ERROR "gcc version must be at least 4.9")
    endif ( )
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    # options for icc: also needs -std=c11
    set (CMAKE_C_FLAGS  "-std=c11 -fopenmp")
    if (CMAKE_C_COMPILER_VERSION VERSION_LESS 18.0)
        message (FATAL_ERROR "icc version must be at least 18.0")
    endif ( )
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    # options for clang
    if (CMAKE_C_COMPILER_VERSION VERSION_LESS 3.3)
        message (FATAL_ERROR "clang version must be at least 3.3")
    endif ( )
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    # options for MicroSoft Visual Studio
endif ( )

# create the dynamic graphblas library.  Requires ANSI C11
file ( GLOB GRAPHBLAS_SOURCES "Source/*.c" "Source/Generated/*.c" )
add_library ( graphblas SHARED ${GRAPHBLAS_SOURCES} )
SET_TARGET_PROPERTIES ( graphblas PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    C_STANDARD_REQUIRED 11
    PUBLIC_HEADER "Include/GraphBLAS.h" )
set_property ( TARGET graphblas PROPERTY C_STANDARD 11 )
target_link_libraries ( graphblas m )

# create the static graphblas library.  Requires ANSI C11
add_library ( graphblas_static STATIC ${GRAPHBLAS_SOURCES} )
SET_TARGET_PROPERTIES ( graphblas_static PROPERTIES
    VERSION ${GraphBLAS_VERSION_MAJOR}.${GraphBLAS_VERSION_MINOR}.${GraphBLAS_VERSION_SUB}
    OUTPUT_NAME graphblas
    POSITION_INDEPENDENT_CODE OFF
    SOVERSION ${GraphBLAS_VERSION_MAJOR}
    C_STANDARD_REQUIRED 11
    PUBLIC_HEADER "Include/GraphBLAS.h" )
set_property ( TARGET graphblas_static PROPERTY C_STANDARD 11 )

# Notes from Sebastien Villemot (sebastien@debian.org):
# SOVERSION policy: if a binary compiled against the old version of the shared
# library needs recompiling in order to work with the new version, then a
# SO_VERSION increase # is needed. Otherwise not.  Examples of the changes that
# require a SO_VERSION increase:
#
#   - a public function or static variable is removed
#   - the prototype of a public function changes
#   - the integer value attached to a public #define or enum changes
#   - the fields of a public structure are modified
#
# Examples of changes that do not require a SO_VERSION increase:
#
#   - a new public function or static variable is added
#   - a private function or static variable is removed or modified
#   - changes in the internals of a structure that is opaque to the calling
#       program (i.e. is only a pointer manipulated through public functions of
#       the library)
#   - a public enum is extended (by adding a new item at the end, but without
#       changing the already existing items)

# graphblas installation location
install ( TARGETS graphblas graphblas_static
    LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR} )

# Demo library
file ( GLOB DEMO_SOURCES "Demo/Source/*.c" )
add_library ( graphblasdemo SHARED ${DEMO_SOURCES} )
SET_TARGET_PROPERTIES ( graphblasdemo PROPERTIES
    C_STANDARD_REQUIRED 11 )
set_property ( TARGET graphblasdemo PROPERTY C_STANDARD 11 )

add_library ( graphblasdemo_static STATIC ${DEMO_SOURCES} )
SET_TARGET_PROPERTIES ( graphblasdemo_static PROPERTIES
    C_STANDARD_REQUIRED 11 )
set_property ( TARGET graphblasdemo_static PROPERTY C_STANDARD 11 )

target_link_libraries ( graphblasdemo         m graphblas )
target_link_libraries ( graphblasdemo_static  graphblas_static )

# Demo programs
add_executable ( bfs_demo      "Demo/Program/bfs_demo.c" )
add_executable ( tri_demo      "Demo/Program/tri_demo.c" )
add_executable ( mis_demo      "Demo/Program/mis_demo.c" )
add_executable ( complex_demo  "Demo/Program/complex_demo.c" )
add_executable ( kron_demo     "Demo/Program/kron_demo.c" )
add_executable ( simple_demo   "Demo/Program/simple_demo.c" )
add_executable ( wildtype_demo "Demo/Program/wildtype_demo.c" )

# Libraries required for Demo programs
target_link_libraries ( bfs_demo      graphblas graphblasdemo )
target_link_libraries ( tri_demo      graphblas graphblasdemo )
target_link_libraries ( mis_demo      graphblas graphblasdemo )
target_link_libraries ( complex_demo  graphblas graphblasdemo )
target_link_libraries ( kron_demo     graphblas graphblasdemo )
target_link_libraries ( simple_demo   graphblasdemo )
target_link_libraries ( wildtype_demo graphblas )

