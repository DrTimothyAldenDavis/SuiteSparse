function x = spex_backslash(A, b, option)
%SPEX_BACKSLASH solve Ax=b via sparse integer-preserving factorization.
% spex_backslash computes the exact solution to the sparse linear system
% Ax = b where A and b are stored as doubles. A must be stored as a sparse
% matrix. b must be stored as a set of dense right hand side vectors (b can be
% either 1 or multiple vector(s)).  SPEX compues the result, x, exactly in
% arbitrary-precision rational numbers. The solution x can be returned in the
% following types:
%
% (a) [floating-poing double] - This final rational-to-double conversion means
%   that x may no longer exactly solve Ax = b.
% (b) [vpa matrix] - This is the arbitrary precision type in MATLAB.
% (c) [cell array of strings] - x{i} = 'numerator/denominator', where the
%   numerator and denominator are strings of decimal digits of arbitrary
%   length.
%
% If A is SPD, an exact up-looking LDL factorization is applied. Otherwise,
% an exact left-looking LU factorization is applied.
%
% Usage:
%
% x = spex_backslash(A, b) returns the solution to Ax = b using default settings.
%
% x = spex_backslash(A, b, options) returns the solution to Ax = b with user
%   defined settings in an options struct.  Entries not present are treated as
%   defaults.
%
%   option.order: Column ordering used.
%       'default' (or if not present): use COLAMD for LU, or use AMD for
%           Cholesky or LDL.
%       'none': no column ordering; factorize the matrix A as-is
%       'colamd': COLAMD
%       'amd': AMD
%
%   option.pivot: Row pivoting scheme used if LU factorization is used.
%       'smallest': Smallest pivot (default)
%       'diagonal': Diagonal pivoting
%       'first': First nonzero per column chosen as pivot
%       'tol smallest': Diagonal pivoting with tol for smallest pivot
%       'tol largest': Diagonal pivoting with tol for largest pivot
%       'largest': Largest pivot
%
%   option.tol: tolerance (0,1] for 'tol smallest' or 'tol largest' pivoting.
%       default is 1.  Only applicable if LU factorization is used.
%
%   option.print: display the inputs and outputs
%       0: nothing (default), 1: just errors, 2: terse, 3: all
%
%   option.solution: a string determining how x is to be returned:
%       'double':  x is converted to a 64-bit floating-point approximate
%           solution.  This is the default.
%       'vpa':  x is returned as a vpa array with option.digits digits (default
%           is given by the MATLAB digits function).  The result may be
%           inexact, if an entry in x cannot be exactly represented in the
%           specified number of digits.  The conversion from the SPEX
%           exact solution (stored as a rational vector) to an arbitrary
%           precision vpa number is very slow (it can be much slower than
%           exactly solving the system Ax = b).
%       'char':  x is returned as a cell array of strings, where
%           x {i} = 'numerator/denominator' and both numerator and denominator
%           are arbitrary-length strings of decimal digits.  The result is
%           always exact, although x cannot be directly used in MATLAB for
%           numerical calculations.  It can be inspected or analyzed using
%           MATLAB string manipulation. Within MATLAB, x may be conversted to
%           vpa, (x=vpa(x)), and then to double (x=double(vpa(x))).
%
%   option.digits: the number of decimal digits to use for x, if
%       option.solution is 'vpa'.  Must be in range 2 to 2^29.
%
% Example:
%
%   % In this first example, x = spex_backslash(A, b) returns an approximate
%   % solution. Since SPEX computes the solution exactly, the only source of
%   % round-of-errors is the final rational-to-double conversion.
%
%   load west0479
%   A = west0479 ;
%   n = size (A, 1) ;
%   xtrue = rand (n,1) ;
%   b = A*xtrue ;
%   x = spex_backslash (A, b) ;
%   err = norm (x-xtrue)
%   x = A\b ;
%   err = norm (x-xtrue)
%
%   % In this example, x = spex_backslash(A,b) is returned exactly in the
%   % MATLAB vector x, because x contains only integers representable exactly
%   % in double precision.
%   % In contrast using x = A\b results in floating-point roundoff error.
%
%   amax = max (abs (A), [ ], 'all') ;
%   A = floor (2^20 * (A / amax)) + n * speye (n) ;
%   xtrue = floor (64 * xtrue) ;
%   b = A*xtrue ;
%   x = spex_backslash (A, b) ;
%   % error and residual will be exactly zero:
%   err = norm (x-xtrue)
%   resid = norm (A*x-b)
%   x = A\b ;
%   % error and residual will be nonzero:
%   err = norm (x-xtrue)
%   resid = norm (A*x-b)
%
% See also vpa, spex_lu_backslash, spex_cholesky_backslash, spex_ldl_backslash,
% spex_mex_install, spex_mex_test, spex_mex_demo.

% spex_backslash is a wrapper for the exact routines contained within the SPEX
% software package.  In order to use spex_backslash you must install the MATLAB
% interfaces of all SPEX packages.  Typing spex_mex_install in this directory
% should do this correctly.

% Copyright (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
% Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

if (nargin < 3)
    option = [ ];   % use default options
end

if (~isnumeric(A) || ~isnumeric(b))
    error ('inputs must be numeric');
end

% SPEX Backslash expects sparse input.
% So, if A is not sparse it is sprasified.
if (~issparse(A))
    A = sparse(A);
end

% Preprocessing complete. Now use SPEX Backslash to solve Ax = b.
x = spex_backslash_mex_soln(A, b, option);

% At this point, depending on the user options, x is either
% (a) a 64-bit floating-point (i.e., double) approximate solution
% (in this case the only source of roundoff errors was the final
% conversion from a rational number to a double), or
% (b) a cell array of strings, where x {i} = 'numerator/denominator'
% and both numerator and denominator are arbitrary-length strings of decimal digits.

% if requested, convert to vpa.
% if provided, use the requested number of digits (otherwise use
% the default in MATLAB).
if (isfield(option, 'solution') && isequal(option.solution, 'vpa'))
    if (isfield(option, 'digits'))
        x = vpa(x, option.digits);
    else
        % use the current default # of digits for vpa.  The default is 32,
        % but this can be changed as a global setting, by the MATLAB 'digits'
        % command.
        x = vpa(x);
    end
end

