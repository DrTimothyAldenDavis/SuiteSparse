#-------------------------------------------------------------------------------
# LAGraph/src/CMakeLists.txt:  cmake script for LAGraph/src
#-------------------------------------------------------------------------------

# LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-2-Clause
#
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM22-0790

#-------------------------------------------------------------------------------

include_directories ( ${CMAKE_SOURCE_DIR}/include utility )

file ( GLOB LAGRAPH_LIB_SOURCES "utility/*.c" "algorithm/*.c" )

if ( WIN32 )
    set ( M_LIB "" )
else ( )
    set ( M_LIB "m" )
endif ( )

#-------------------------------------------------------------------------------
# dynamic lagraph library properties
#-------------------------------------------------------------------------------

add_library ( lagraph SHARED ${LAGRAPH_LIB_SOURCES} )
SET_TARGET_PROPERTIES ( lagraph PROPERTIES
    VERSION ${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}
    SOVERSION ${LAGraph_VERSION_MAJOR}
    C_STANDARD_REQUIRED ON
    C_STANDARD 11
    PUBLIC_HEADER "include/LAGraph.h"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dlls )
target_link_libraries ( lagraph PRIVATE ${GRAPHBLAS_LIBRARIES} ${M_LIB} )
target_link_directories ( lagraph BEFORE PUBLIC ${CMAKE_SOURCE_DIR}/build )
target_include_directories ( lagraph PUBLIC include/ )
target_compile_definitions ( lagraph PRIVATE LG_LIBRARY )
target_compile_definitions ( lagraph PUBLIC LG_DLL )
# install ( FILES ${LAGRAPH_INCLUDE} DESTINATION include)

#-------------------------------------------------------------------------------
# static lagraph library properties
#-------------------------------------------------------------------------------

add_library ( lagraph_static STATIC ${LAGRAPH_LIB_SOURCES} )
SET_TARGET_PROPERTIES ( lagraph_static PROPERTIES
    VERSION ${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}
    OUTPUT_NAME lagraph
    POSITION_INDEPENDENT_CODE OFF
    SOVERSION ${LAGraph_VERSION_MAJOR}
    C_STANDARD_REQUIRED ON
    C_STANDARD 11 )

if ( MSVC )
    set_target_properties ( lagraph_static PROPERTIES
        OUTPUT_NAME lagraph_static )
endif ( )

target_link_libraries ( lagraph_static PUBLIC ${GRAPHBLAS_LIBRARIES} ${M_LIB} )
target_link_directories( lagraph_static BEFORE PUBLIC ${CMAKE_SOURCE_DIR}/build )

#-------------------------------------------------------------------------------
# add OpenMP
#-------------------------------------------------------------------------------

if ( OPENMP_C_FOUND )
    target_link_libraries ( lagraph PRIVATE OpenMP::OpenMP_C )
    target_link_libraries ( lagraph_static PUBLIC OpenMP::OpenMP_C )
endif ( )

#-------------------------------------------------------------------------------
# applications
#-------------------------------------------------------------------------------

add_subdirectory ( test )
add_subdirectory ( benchmark )

