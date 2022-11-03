function x = SPEX_Left_LU_backslash (A,b,option)
% SPEX_Left_LU_BACKSLASH: solve Ax=b via sparse left-looking integer-preserving LU
% SPEX_Left_LU_backslash: computes the exact solution to the sparse linear system Ax =
% b where A and b are stored as doubles. A must be stored as a sparse matrix. b
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
% x = SPEX_Left_LU_backslash (A,b) returns the solution to Ax=b using default settings.
%
% x = SPEX_Left_LU_backslash (A,b,options) returns the solution to Ax=b with user
%   defined settings in an options struct.  Entries not present are treated as
%   defaults.
%
%   option.order: Column ordering used.
%       'none': no column ordering; factorize the matrix A as-is
%       'colamd': COLAMD (the default ordering)
%       'amd': AMD
%
%   option.pivot: Row pivoting scheme used.
%       'smallest': Smallest pivot
%       'diagonal': Diagonal pivoting
%       'first': First nonzero per column chosen as pivot
%       'tol smallest': Diagonal pivoting with tol for smallest pivot (default)
%       'tol largest': Diagonal pivoting with tol for largest pivot
%       'largest': Largest pivot
%
%   option.tol: tolerance (0,1] for 'tol smallest' or 'tol largest' pivoting.
%       default is 0.1.
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
%   % In this first example, x = SPEX_backslash (A,b) returns an approximate
%   % solution, not because it was computed incorrectly in SPEX_backslash.  It
%   % is computed exactly as a rational result in SPEX_backslash with arbitrary
%   % precision, but then converted to double precision on output.
%
%   load west0479
%   A = west0479 ;
%   n = size (A, 1) ;
%   xtrue = rand (n,1) ;
%   b = A*xtrue ;
%   x = SPEX_Left_LU_backslash (A, b) ;
%   err = norm (x-xtrue)
%   x = A\b ;
%   err = norm (x-xtrue)
%
%   % In this example, x = SPEX_backslash (A,b) is returned exactly in the
%   % MATLAB vector x, because x contains only integers representable exactly
%   % in double precision.  x = A\b results in floating-point roundoff error.
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
% See also vpa, SPEX_install, SPEX_test, SPEX_demo.

% SPEX_Left_LU_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
% Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
% SPEX_Left_LU/License for the license.

if (nargin < 3)
    option = [ ] ;   % use default options
end

if (~isnumeric (A) || ~isnumeric (b))
    error ('inputs must be numeric') ;
end

% Check if the input matrix is stored as sparse. If not, SPEX Left LU expects
% sparse input, so convert to sparse.
if (~issparse (A))
    A = sparse (A) ;
end

% Preprocessing complete. Now use SPEX Left LU to solve A*x=b.
x = SPEX_Left_LU_mex_soln (A, b, option) ;

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

