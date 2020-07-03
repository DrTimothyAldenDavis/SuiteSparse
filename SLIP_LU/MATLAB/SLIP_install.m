function SLIP_install(run_demo)
%SLIP_INSTALL: install and test the MATLAB interface to SLIP_backslash.
% This function installs the SLIP LU mexFunction for use by the m-file
% SLIP_backslash.m.
%
% Usage: SLIP_install
%
% Required Libraries: GMP, MPFR, AMD, COLAMD.  If -lamd and -lcolamd are not
% available, install them with 'make install' first, in the top-level
% SuiteSparse folder.
%
% See also SLIP_backslash, SLIP_test, SLIP_demo.

% SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
% Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
% SLIP_LU/License for the license.

if (nargin < 1)
    run_demo = true ;
end

fprintf ('Compiling the SLIP LU mexFunction for use in SLIP_backslash:\n') ;

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

% Compiler flags
flags = 'CFLAGS=''-std=c99 -fPIC''';

% External libraries: GMP, MPRF, AMD, and COLAMD
libs = '-L../../lib -lgmp -lmpfr -lamd -lcolamd -lsuitesparseconfig' ;

% Path to headers
includes = '-ISource/ -I../Source/ -I../Include/ -I../../SuiteSparse_config -I../../COLAMD/Include -I../../AMD/Include';

% verbose = ' -v '
verbose = '' ;

% Generate the mex commands here
% having -R2018a here for function mxGetDoubles
m1 = ['mex ', verbose, ' -R2018a ', includes, ' SLIP_mex_soln.c ' , src, ' ', flags, ' ', libs];

if (~isempty (verbose))
    fprintf ('%s\n', m1) ;
end

% Now, we evaluate each one
eval (m1) ;

if (run_demo)
    % Test SLIP_backslash.
    SLIP_test ;
end

fprintf ('To use SLIP_backslash in future MATLAB sessions, add the following\n') ;
fprintf ('line to your startup.m file:\n') ;
fprintf ('   addpath (''%s'') ;\n', pwd) ;
fprintf ('Type ''doc startup'' for more info on how to use startup.m\n') ;
fprintf ('To run a demo, type:\n') ;
fprintf ('   echodemo SLIP_demo ;\n') ;

