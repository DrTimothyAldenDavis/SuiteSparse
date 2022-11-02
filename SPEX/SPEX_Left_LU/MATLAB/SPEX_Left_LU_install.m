function SPEX_Left_LU_install(run_demo)
% SPEX_Left_LU_INSTALL: install and test the MATLAB interface to SPEX_backslash.
% This function installs the SPEX Left LU mexFunction for use by the m-file
% SPEX_Left_LU_backslash.m.
%
% Usage: SPEX_Left_LU_install
%
% Required Libraries: GMP, MPFR, AMD, COLAMD.  If -lamd and -lcolamd are not
% available, install them with 'make install' first, in the top-level
% SuiteSparse/lib folder, and add SuiteSparse/lib to your LD_LIBRARY_PATH
% and restart MATLAB.
%
% See also SPEX_Left_LU_backslash, SPEX_Left_LU_test, SPEX_Left_LU_demo.

% SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
% Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

help SPEX_Left_LU_install

if (nargin < 1)
    run_demo = true ;
end

fprintf ('Compiling the SPEX Left LU mexFunction for use in SPEX_Left_LU_backslash:\n') ;

% Find all source files and add them to the src string
src = '';
path = './Source/';
files = dir('./Source/*.c');
[m n] = size(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end
path = '../Source/';
files = dir('../Source/*.c');
[m n] = size(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end
path = '../../SPEX_Util/Source/';
files = dir('../../SPEX_Util/Source/*.c');
[m n] = size(files);
for k = 1:m
    tmp = [' ', path, files(k).name];
    src = [src, tmp];
end

% Compiler flags
flags = 'CFLAGS=''-std=c99 -fPIC''';

% External libraries: GMP, MPRF, AMD, and COLAMD
libs = '-L../../../lib -lgmp -lmpfr -lamd -lcolamd -lsuitesparseconfig' ;

% Path to headers
includes = '-ISource/ -I../Source/ -I../../Include -I../../../SuiteSparse_config -I../../../COLAMD/Include -I../../../AMD/Include -I../../SPEX_Util/Source';

% verbose = ' -v '
verbose = '' ;

% Generate the mex commands here
% having -R2018a here for function mxGetDoubles
m1 = ['mex ', verbose, ' -R2018a ', includes, ' SPEX_Left_LU_mex_soln.c ' , src, ' ', flags, ' ', libs];

if (~isempty (verbose))
    fprintf ('%s\n', m1) ;
end

% Now, we evaluate each one
eval (m1) ;

if (run_demo)
    % Test SPEX_backslash.
    SPEX_Left_LU_test ;
end

fprintf ('To use SPEX_Left_LU_backslash in future MATLAB sessions, add the following\n') ;
fprintf ('line to your startup.m file:\n') ;
fprintf ('   addpath (''%s'') ;\n', pwd) ;
fprintf ('Type ''doc startup'' for more info on how to use startup.m\n') ;
fprintf ('To run a demo, type:\n') ;
fprintf ('   echodemo SPEX_Left_LU_demo ;\n') ;

