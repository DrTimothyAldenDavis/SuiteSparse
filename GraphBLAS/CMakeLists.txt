# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
# http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.
#
# CMakeLists.tx: instructions for cmake to build GraphBLAS.
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

# cmake 3.0 is prefered.
cmake_minimum_required ( VERSION 2.8.12 )

project ( graphblas )

if ( CMAKE_VERSION VERSION_GREATER "3.0" )
    cmake_policy ( SET CMP0042 NEW )
endif ( )

if (NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

# For development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

# include directories for both graphblas and graphblasdemo libraries
include_directories ( Source/Template Include Demo/Include )

# check which compiler is being used.  If you need to make
# compiler-specific modifications, here is the place to do it.
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # cmake 2.8 workaround: gcc needs to be told to do ANSI C11.
    # cmake 3.0 doesn't have this problem.
    set (CMAKE_C_FLAGS  "-std=c11 -lm")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    # options for icc: also needs -std=c11
    set (CMAKE_C_FLAGS  "-std=c11")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    # options for clang
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    # options for MicroSoft Visual Studio
endif ( )

# create the graphblas library.  Requires ANSI C11
file ( GLOB GRAPHBLAS_SOURCES "Source/*.c" )
add_library ( graphblas SHARED ${GRAPHBLAS_SOURCES} )
SET_TARGET_PROPERTIES ( graphblas PROPERTIES VERSION 1.0.0
    SOVERSION 1
    C_STANDARD_REQUIRED 11
    PUBLIC_HEADER "Include/GraphBLAS.h" )
set_property ( TARGET graphblas PROPERTY C_STANDARD 11 )

# graphblas installation location
install ( TARGETS graphblas
    LIBRARY       DESTINATION /usr/local/lib
    PUBLIC_HEADER DESTINATION /usr/local/include )

# Demo library
file ( GLOB DEMO_SOURCES "Demo/Source/*.c" )
add_library ( graphblasdemo SHARED ${DEMO_SOURCES} )
SET_TARGET_PROPERTIES ( graphblasdemo PROPERTIES
    C_STANDARD_REQUIRED 11 )
set_property ( TARGET graphblasdemo PROPERTY C_STANDARD 11 )

target_link_libraries ( graphblasdemo  graphblas )

# Demo programs
add_executable ( bfs_demo      "Demo/Program/bfs_demo.c" )
add_executable ( tri_demo      "Demo/Program/tri_demo.c" )
add_executable ( mis_demo      "Demo/Program/mis_demo.c" )
add_executable ( complex_demo  "Demo/Program/complex_demo.c" )
add_executable ( simple_demo   "Demo/Program/simple_demo.c" )
add_executable ( wildtype_demo "Demo/Program/wildtype_demo.c" )

# Libraries required for Demo programs
target_link_libraries ( bfs_demo      graphblas graphblasdemo )
target_link_libraries ( tri_demo      graphblas graphblasdemo )
target_link_libraries ( mis_demo      graphblas graphblasdemo )
target_link_libraries ( complex_demo  graphblas graphblasdemo )
target_link_libraries ( simple_demo   graphblasdemo )
target_link_libraries ( wildtype_demo graphblas )

