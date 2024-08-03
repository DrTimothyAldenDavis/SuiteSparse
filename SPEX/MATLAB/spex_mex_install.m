function spex_mex_install(run_demo)
%SPEX_MEX_INSTALL install and test the MATLAB interface to SPEX MATLAB
% functions.
%
% Usage:
%   spex_mex_install            % compile the mexFunctions and run the tests
%   spex_mex_install (0)        % do not run the the tests after installation
%
% Required Libraries: GMP and MPFR.  You must run cmake in the top-level SPEX
% folder first, to configure the spex_deps.m file so that this installation
% script can find the GMP and MPRF libraries.
%
% See also spex_deps, spex_demo.

% Copyright (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
% Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

%#ok<*AGROW>

if (nargin < 1)
    run_demo = true ;
end

fprintf ('Compiling the SPEX for use in MATLAB:\n') ;

% Find all source files and add them to the src string
src = '';
path = './Source/';
files = dir('./Source/*.c');
m = length(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end
path = '../SPEX_Utilities/Source/';
files = dir('../SPEX_Utilities/Source/*.c');
m = length(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end

path = '../SPEX_LU/Source/';
files = dir('../SPEX_LU/Source/*.c');
m = length(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end

path = '../SPEX_Cholesky/Source/';
files = dir('../SPEX_Cholesky/Source/*.c');
m = length(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end

path = '../SPEX_Backslash/Source/';
files = dir('../SPEX_Backslash/Source/*.c');
m = length(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end

path = '../../AMD/Source/';
files = dir('../../AMD/Source/amd_l*.c');
m = length(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end

src = [src ' ../../COLAMD/Source/colamd_l.c' ] ;
src = [src ' ../../COLAMD/Source/colamd_version.c' ] ;
src = [src ' ../../SuiteSparse_config/SuiteSparse_config.c' ] ;

% External libraries: GMP, MPRF, AMD, and COLAMD
[gmp_lib, gmp_include, mpfr_lib, mpfr_include] = spex_deps ;

% Compiler flags
if (ismac)
    flags = 'CFLAGS=''-std=c11 -DCLANG_NEEDS_MAIN=1 -fPIC ''' ;
    flags = [flags ' -DCLANG_NEEDS_MAIN'] ;
else
    flags = 'CFLAGS=''-std=c11 -fPIC ''' ;
end

flags = [' -O ' flags ] ;

% libraries:
libs = [gmp_lib ' ' mpfr_lib ' -lm'] ;

% Path to headers
if (isempty (gmp_include))
    gmp_include = ' ' ;
else
    gmp_include = [' -I' gmp_include ' '] ;
end
if (isempty (mpfr_include))
    mpfr_include = ' ' ;
else
    mpfr_include = [' -I' mpfr_include ' '] ;
end

includes = ' -ISource/ -I../Include/ -I../SPEX_Utilities/Source ' ;
includes = [includes gmp_include  mpfr_include ] ;
includes = [includes ' -I../../AMD/Source  -I../../AMD/Include  '] ;
includes = [includes ' -I../../COLAMD/Source  -I../../COLAMD/Include  '] ;
includes = [includes ' -I../../SuiteSparse_config  '] ;

% verbose = ' -v '
verbose = '' ;

% Generate the mex commands here
% having -R2018a here for function mxGetDoubles
m1 = ['mex ', verbose, ' -R2018a ', includes, ' spex_lu_mex_soln.c ' , src, ' ', flags, ' ', libs] ;
m2 = ['mex ', verbose, ' -R2018a ', includes, ' spex_cholesky_mex_soln.c ' , src, ' ', flags, ' ', libs];
m3 = ['mex ', verbose, ' -R2018a ', includes, ' spex_ldl_mex_soln.c ' , src, ' ', flags, ' ', libs];
m4 = ['mex ', verbose, ' -R2018a ', includes, ' spex_backslash_mex_soln.c ' , src, ' ', flags, ' ', libs];

% Now, we evaluate each one
if (~isempty (verbose))
    fprintf ('%s\n', m1) ;
end
fprintf ('Compiling MATLAB interface to SPEX LU (please wait):\n') ;
eval (m1) ;

if (~isempty (verbose))
    fprintf ('%s\n', m2) ;
end
fprintf ('Compiling MATLAB interface to SPEX Cholesky (please wait):\n') ;
eval (m2) ;

if (~isempty (verbose))
    fprintf ('%s\n', m3) ;
end
fprintf ('Compiling MATLAB interface to SPEX LDL (please wait):\n') ;
eval (m3) ;

if (~isempty (verbose))
    fprintf ('%s\n', m4) ;
end
fprintf ('Compiling MATLAB interface to SPEX Backslash (please wait):\n') ;
eval (m4) ;

if (run_demo)
    % Test SPEX
    spex_mex_test ;
end

fprintf ('To use SPEX MATLAB Interface in future MATLAB sessions, add the following\n') ;
fprintf ('line to your startup.m file:\n') ;
fprintf ('   addpath (''%s'') ;\n', pwd) ;
fprintf ('Type ''doc startup'' for more info on how to use startup.m\n') ;
fprintf ('To run a demo, type:\n') ;
fprintf ('   spex_demo\n') ;


