#-------------------------------------------------------------------------------
# GraphBLAS/JITpackage/CMakeLists.txt:  package source for the GraphBLAS JIT
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.20 ) # GraphBLAS can be built stand-alone
project ( jitpackage LANGUAGES C )
add_executable ( grb_jitpackage "Source/grb_jitpackage.c" )
target_include_directories ( grb_jitpackage PRIVATE ../zstd/zstd_subset )
if ( NOT WIN32 )
    target_link_libraries ( grb_jitpackage PRIVATE m )
endif ( )

file ( GLOB GRB_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "../Include/GraphBLAS.h"
    "../Source/Template/*.[ch]"
    "../Source/JitKernels/*.[ch]"
    "../CUDA/Template/*h"
    "../CUDA/JitKernels/*h"
    "../Source/Shared/*.h" )

add_custom_command ( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/GB_JITpackage.c
    COMMAND $<TARGET_FILE:grb_jitpackage>
    ARGS ${GRB_SOURCE_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${GRB_SOURCE_FILES}
    COMMENT "Generating compressed sources for JIT compiler..."
)

add_custom_target ( GB_JITpackage
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/GB_JITpackage.c )
