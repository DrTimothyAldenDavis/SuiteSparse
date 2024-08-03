%% a demo of the SPEX MATLAB interface
% SPEX is a package for solving sparse linear systems of
% equations with a roundoff-free integer-preserving method.
% The result is always exact, unless the matrix A is perfectly
% singular.
%
% See also vpa, spex_backslash, spex_lu_backslash,
%   spex_cholesky_backslash, spex_ldl_backslash,
%   spex_mex_install, spex_mex_test.
%
% Copyright (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
% Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
% All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

%#ok<*NOPTS>
%#ok<*NASGU>

%% SPEX vs MATLAB backslash: first example
% In this first example, x = spex_backslash (A, b) returns an
% approximate solution, but not because it was computed
% incorrectly in spex_backslash.  It is computed exactly as a
% rational result in spex_backslash with arbitrary precision,
% but then converted to double precision on output.

format long g
load west0479
A = west0479 ;
n = size (A, 1) ;
xtrue = rand (n,1) ;
b = A*xtrue ;
x = spex_backslash (A, b) ;
% error is nonzero: x is computed exactly in rational
% arbitrary-precision, but then loses precision when
% returned as a double vector to MATLAB:
err_spex = norm (x-xtrue)

x = A\b ;
% error is nonzero: MATLAB x=A\b experiences floating-point
% error throughout its computations:
err_matlab = norm (x-xtrue)

%% SPEX backslash: exact, vs MATLAB backslash: approximate
% In this example, x = spex_backslash (A, b) is returned
% exactly in the MATLAB vector x, because x contains only
% integers representable exactly in double precision.  x = A\b
% results in floating-point roundoff error.

amax = max (abs (A), [ ], 'all') ;
A = floor (2^20 * (A / amax)) + n * speye (n) ;
xtrue = floor (64 * xtrue) ;
b = A*xtrue ;
x = spex_backslash (A, b) ;
% error will be exactly zero:
err_spex = norm (x-xtrue)
x = A\b ;
% error will be small but nonzero:
err_matlab = norm (x-xtrue)

%% SPEX Backslash on singular problems
% SPEX correctly determines that the following integer matrix is
% singular.  The MATLAB numerical Cholesky factorization fails to
% do so, returning a useless solution.

A = sparse ([ % T = JGD_Homology/ch3-3-b1 matrix ; A=T'*T:
      4    -1    -1     0    -1     0     0    -1     0
     -1     4     0    -1     0    -1     0    -1     0
     -1     0     4    -1    -1     0    -1     0     0
      0    -1    -1     4     0    -1    -1     0     0
     -1     0    -1     0     4    -1     0     0    -1
      0    -1     0    -1    -1     4     0     0    -1
      0     0    -1    -1     0     0     4    -1    -1
     -1    -1     0     0     0     0    -1     4    -1
      0     0     0     0    -1    -1    -1    -1     4 ]) ;
b = ones (9,1);
try
    x = spex_backslash (A, b) ;
catch me
    fprintf ("\nSPEX Error: %s\n", me.message) ;
end

R = chol(A);
x_chol_matlab = R \ (R' \ b)

%% SPEX on an ill-conditioned problem
% x = spex_backslash (A,b) is able to accurately solve problems
% that x=A\b cannot.  Consider the Wilkinson gallery matrix:

[A, b] = gallery ('wilk', 3)

%% vpa can find a good but not perfect solution:
xvpa = vpa (A) \ b

% but MATLAB's numerical x = A\b computes a poor solution:
xapprox = A \ b

%% spex_backslash computes the exact answer
% SPEX returns the exact solution a double vector:

xspex = spex_backslash (A, b)
err = xvpa - xspex
relerr = double (err (2:3) ./ xvpa (2:3))

%% spex_backslash with exact results
% spex_backslash can also return x as a cell array of strings,
% which preserves the exact rational result.  The printing option
% is also enabled in this example.  The floating-point matrices A
% and b are converted into a scaled integer matrix before solving
% A*x=b with SPEX Left LU.
%
% SPEX obtains an integer matrix by scaling the input. SPEX
% scales all input by 1e16. This is because consider the number
% A(1,2). The value A(1,2)=0.9 is a floating point number and
% cannot be represented exactly in IEEE floating-point.
% Specifically, the rational represenation of it is fl(0.9) =
% 45000000000000001 / 50000000000000000.
%
% SPEX assumes the user wants what they typed in. Scaling this
% matrix exactly gives the above rational. Conversely scaling
% with 16 digits of precision gives the exact fraction 9/10.
%
% If one wishes to obtain FULL floating-point precision and/or
% support for floating point values smaller than 1e-16 there are
% two options:
%
%   1) Within MATLAB the user scales the matrix themselves. If
%      SPEX is given an integer matrix it is preserved exactly.
%
%   2) Within C convert the matrix to a SPEX_MPFR. MPFR numbers
%      can exactly store doubles. The conversion from MPFR to
%      integer is fully exact and all one would obtain the
%      rational representation of the floating-point number
%      itself. This can be done with 1 call to the SPEX matrix
%      copy function.
%
% In any case, below is the exact solution with A(1,2) = 9/10

option.print = 3 ;          % also print the details
option.solution = 'char' ;  % return x as a cell array of strings

%%

xspex = spex_backslash (A, b, option)

%% Converting an exact rational result to vpa or double
% If spex_backslash returns x as a cell array of strings, it
% cannot be immediately used in computations in MATLAB.  It can
% be converted into a vpa or double matrix, as illustrated below.

xspex_as_vpa = vpa (xspex)
xspex_as_double = double (vpa (xspex))
xvpa_as_double = double (xvpa)

%% Comparing the VPA and SPEX_BACKSLASH solutions in double
% Both vpa(A)\b and spex_backslash(A,b) compute the same result
% in the end, when their results are converted to double.
err = xvpa_as_double - xspex_as_double






