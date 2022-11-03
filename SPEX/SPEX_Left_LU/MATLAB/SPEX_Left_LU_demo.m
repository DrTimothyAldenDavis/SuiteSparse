% SPEX_Left_LU_DEMO a demo of SPEX_Left_LU_backslash
% SPEX_Left_LU_LU is a package for solving sparse linear systems of equations
% with a roundoff-free integer-preserving method.  The result is
% always exact, unless the matrix A is perfectly singular.
%
% See also vpa, SPEX_Left_LU_backslash, SPEX_Left_LU_install, SPEX_Left_LU_test.
%
% SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
% Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

format compact

%% SPEX_Left_LU_backslash vs MATLAB backslash: first example
% In this first example, x = SPEX_Left_LU_backslash (A,b) returns an approximate
% solution, but not because it was computed incorrectly in SPEX_Left_LU_backslash.
% It is computed exactly as a rational result in SPEX_backslash with
% arbitrary precision, but then converted to double precision on output.

format long g
load west0479
A = west0479 ;
n = size (A, 1) ;
xtrue = rand (n,1) ;
b = A*xtrue ;
x = SPEX_Left_LU_backslash (A, b) ;
% error is nonzero: x is computed exactly in rational arbitrary-precision,
% but then lost precision when returned to MATLAB:
err_spex = norm (x-xtrue)
x = A\b ;
% error is nonzero: MATLAB x=A\b experiences floating-point error
% throughout its computations:
err_matlab = norm (x-xtrue)

%% SPEX_Left_LU_backslash: exact, vs MATLAB backslash: approximate
% In this example, x = SPEX_Left_LU_backslash (A,b) is returned exactly in the
% MATLAB vector x, because x contains only integers representable exactly
% in double precision.  x = A\b results in floating-point roundoff error.

amax = max (abs (A), [ ], 'all') ;
A = floor (2^20 * (A / amax)) + n * speye (n) ;
xtrue = floor (64 * xtrue) ;
b = A*xtrue ;
x = SPEX_Left_LU_backslash (A, b) ;
% error will be exactly zero:
err_spex = norm (x-xtrue)
x = A\b ;
% error will be small but nonzero:
err_matlab = norm (x-xtrue)

%% SPEX_Left_LU_backslash on ill-conditioned problems
% x = SPEX_Left_LU_backslash (A,b) is able to solve problems that x=A\b cannot.
% Consider the following matrix in the MATLAB gallery:

[U, b] = gallery ('wilk', 3)

%%    vpa can find a good but not perfect solution:
xvpa = vpa (U) \ b

%     but MATLAB's numerical x = U\b computes a poor solution:
xapprox = U \ b

%% SPEX_Left_LU_backslash computes the exact answer
% It returns it to MATLAB as a double vector, obtaining the exact results,
% except for a final floating-point error in x(2):

xspex = SPEX_Left_LU_backslash (U, b)
err = xvpa - xspex
relerr = double (err (2:3) ./ xvpa (2:3))

%% SPEX_Left_LU_backslash with exact results
% SPEX_Left_LU_backslash can also return x as a cell array of strings, which
% preserves the exact rational result.  The printing option is also
% enabled in this example.  The floating-point matrices U and b are
% converted into a scaled integer matrix before solving U*x=b with
% SPEX Left LU.
%
% The value U(1,2)=0.9 is a floating-point number, and 0.9 cannot be
% exactly represented in IEEE floating-point representation.  It is
% converted exactly into the rational number,
% fl(0.9) = 45000000000000001 / 50000000000000000.

option.print = 3 ;          % also print the details
option.solution = 'char' ;  % return x as a cell array of strings

%%

xspex = SPEX_Left_LU_backslash (U, b, option)

%% Converting an exact rational result to vpa or double
% If SPEX_backslash returns x as a cell array of strings, it cannot
% be immediately used in computations in MATLAB.  It can be converted
% into a vpa or double matrix, as illustrated below.  The solution
% differs slightly from the vpa solution xvpa = vpa (U)\b, since
% the MATLAB vpa converts fl(0.9) into a decimal representation 0.9,
% or exactly 9/10; this is not exactly equal to fl(0.9), since the
% value 9/10 is not representable in IEEE floating-point.  SPEX_backslash,
% by contrast, converts fl(0.9) into its exact rational representation,
% 45000000000000001 / 50000000000000000.

xspex_as_vpa = vpa (xspex)
xspex_as_double = double (vpa (xspex))
xvpa_as_double = double (xvpa)

%% Comparing the VPA and SPEX_Left_LU_BACKSLASH solutions in double
% Both vpa(U)\b and SPEX_backslash(U,b) compute the same result
% in the end, when their results are converted to double.
err = xvpa_as_double - xspex_as_double

