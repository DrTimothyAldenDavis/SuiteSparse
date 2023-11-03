#-------------------------------------------------------------------------------
# GraphBLAS/JITpackage/CMakeLists.txt:  package source for the GraphBLAS JIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.16 )
project ( grb_jitpackage LANGUAGES C )
include_directories ( ../zstd ../zstd/zstd_subset )
add_executable ( grb_jitpackage  "Source/grb_jitpackage.c" )
if ( NOT WIN32 )
    target_link_libraries ( grb_jitpackage PUBLIC m )
endif ( )

file ( GLOB GRB_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "../Include/GraphBLAS.h"
    "../Source/Template/*.[ch]"
    "../Source/JitKernels/*.[ch]"
    "../Source/Shared/*.h" )

add_custom_command ( TARGET grb_jitpackage POST_BUILD
    COMMAND $<TARGET_FILE:grb_jitpackage>
    ARGS ${GRB_SOURCE_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating compressed sources for JIT compiler..."
)
