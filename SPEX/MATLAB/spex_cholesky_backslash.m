function x = spex_cholesky_backslash (A,b,option)
% spex_cholesky_BACKSLASH: solve Ax=b via sparse integer-preserving Cholesky
% spex_cholesky_backslash: computes the exact solution to the sparse linear system Ax =
% b where A and b are stored as doubles. A must be stored as a sparse matrix and be SPD. b
% must be stored as a dense set of right hand side vectors. b can be either 1
% or multiple vector(s).  The result x is computed exactly, represented in
% arbitrary-precision rational values, and then returned to MATLAB as a
% floating-poing double result.  This final conversion means that x may no
% longer exactly solve A*x=b, unless this final conversion is able to be
% done without modification.
%
% x may also be returned as a vpa matrix, or a cell array of strings, with
% x {i} = 'numerator/denominator', where the numerator and denominator are
% strings of decimal digits of arbitrary length.
%
% Usage:
%
% x = spex_cholesky_backslash (A,b) returns the solution to Ax=b using default settings.
%
% x = spex_cholesky_backslash (A,b,options) returns the solution to Ax=b with user
%   defined settings in an options struct.  Entries not present are treated as
%   defaults.
%
%   option.order: Column ordering used.
%       'none': no column ordering; factorize the matrix A as-is
%       'colamd': COLAMD (the default ordering)
%       'amd': AMD
%
%   option.print: display the inputs and outputs
%       0: nothing (default), 1: just errors, 2: terse, 3: all
%
%   option.solution: a string determining how x is to be returned:
%       'double':  x is converted to a 64-bit floating-point approximate
%           solution.  This is the default.
%       'vpa':  x is returned as a vpa array with option.digits digits (default
%           is given by the MATLAB digits function).  The result may be
%           inexact, if an entry in x cannot be represented in the specified
%           number of digits.  To convert this x to double, use x=double(x).
%       'char':  x is returned as a cell array of strings, where
%           x {i} = 'numerator/denominator' and both numerator and denominator
%           are arbitrary-length strings of decimal digits.  The result is
%           always exact, although x cannot be directly used in MATLAB for
%           numerical calculations.  It can be inspected or analyzed using
%           MATLAB string manipulation.  To convert x to vpa, use x=vpa(x).  To
%           convert x to double, use x=double(vpa(x)).
%
%   option.digits: the number of decimal digits to use for x, if
%       option.solution is 'vpa'.  Must be in range 2 to 2^29.
%
% Example:
%
%   % In this first example, x = spex_cholesky_backslash (A,b) returns an approximate
%   % solution, not because it was computed incorrectly in spex_cholesky_backslash.  It
%   % is computed exactly as a rational result in SPEX_backslash with arbitrary
%   % precision, but then converted to double precision on output.
%
%   load west0479
%   A = west0479 ;
%   A = A'*A;
%   n = size (A, 1) ;
%   xtrue = rand (n,1) ;
%   b = A*xtrue ;
%   x = spex_cholesky_backslash (A, b) ;
%   err = norm (x-xtrue)
%   x = A\b ;
%   err = norm (x-xtrue)
%
%   % In this example, x = spex_cholesky_backslash (A,b) is returned exactly
%   % in the MATLAB vector x, because x contains only integers representable
%   % exactly in double precision.  x = A\b results in floating-point
%   % roundoff error.
%
%   amax = max (abs (A), [ ], 'all') ;
%   A = floor (2^20 * (A / amax)) + n * speye (n) ;
%   xtrue = floor (64 * xtrue) ;
%   b = A*xtrue ;
%   x = SPEX_backslash (A, b) ;
%   % error and residual will be exactly zero:
%   err = norm (x-xtrue)
%   resid = norm (A*x-b)
%   x = A\b ;
%   % error and residual will be nonzero:
%   err = norm (x-xtrue)
%   resid = norm (A*x-b)
%
% See also vpa, spex_mex_install, spex_mex_test, spex_mex_demo,
%   spex_lu_backslash

% spex_cholesky_backslash is a wrapper for the exact routines contained within the SPEX
% software package.  In order to use spex_cholesky_backslash you must install the MATLAB
% interfaces of all SPEX packages.  Typing spex_mex_install in this directory
% should do this correctly.

% SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
% Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

if (nargin < 3)
    option = [ ];   % use default options
end

if (~isnumeric (A) || ~isnumeric (b))
    error ('inputs must be numeric') ;
end

% Check if the input matrix is stored as sparse. If not, SPEX Chol expects
% sparse input, so convert to sparse.
if (~issparse (A))
    A = sparse (A) ;
end

% Ensure that input appears to be symmetric.
if ( normest(A-A') > 1e-6)
    error('inputs must be symmetric')
end


% Preprocessing complete. Now use SPEX Chol to solve A*x=b.
x=spex_cholesky_mex_soln (A, b, option) ;

% convert to vpa, if requested
if (isfield (option, 'solution') && isequal (option.solution, 'vpa'))
    if (isfield (option, 'digits'))
        x = vpa (x, option.digits) ;
    else
        % use the current default # of digits for vpa.  The default is 32,
        % but this can be changed as a global setting, by the MATLAB 'digits'
        % command.
        x = vpa (x) ;
    end
end

