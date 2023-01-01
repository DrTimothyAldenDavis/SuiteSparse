#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseFortran.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# SuiteSparse CMake policies for Fortran.  The following parameters can be
# defined prior to including this file:

#   SUITESPARSE_C_TO_FORTRAN:  a string that defines how C calls Fortran.
#                       Defaults to "(name,NAME) name" for Windows (lower case,
#                       no underscore appended to the name), which is the
#                       system that is most likely not to have a Fortran
#                       compiler.  Defaults to "(name,NAME) name##_" otherwise.
#                       This setting is only used if no Fortran compiler is
#                       found.
#
#   NFORTRAN:           if true, no Fortan files are compiled, and the Fortran
#                       language is not enabled in any cmake scripts.  The
#                       built-in cmake script FortranCInterface is skipped.
#                       This will require SUITESPARSE_C_TO_FORTRAN to be defined
#                       explicitly, if the defaults are not appropriate for your
#                       system.
#                       Default: false

#-------------------------------------------------------------------------------
# check if Fortran is available and enabled
#-------------------------------------------------------------------------------

option ( NFORTRAN "ON: do not try to use Fortran. OFF (default): try Fortran" off )
if ( NFORTRAN )
    message ( STATUS "Fortran: not enabled" )
else ( )
    include ( CheckLanguage )
    check_language ( Fortran )
    if ( CMAKE_Fortran_COMPILER )
        enable_language ( Fortran )
        message ( STATUS "Fortran: ${CMAKE_Fortran_COMPILER}" )
    else ( )
        # Fortran not available:
        set ( NFORTRAN true )
        message ( STATUS "Fortran: not available" )
    endif ( )
endif ( )

# default C-to-Fortran name mangling if Fortran compiler not found
if ( MSVC )
    # MS Visual Studio Fortran compiler does not mangle the Fortran name
    set ( SUITESPARSE_C_TO_FORTRAN "(name,NAME) name"
        CACHE STRING "C to Fortan name mangling" )
else ( )
    # Other systems (Linux, Mac) typically append an underscore
    set ( SUITESPARSE_C_TO_FORTRAN "(name,NAME) name##_"
        CACHE STRING "C to Fortan name mangling" )
endif ( )

