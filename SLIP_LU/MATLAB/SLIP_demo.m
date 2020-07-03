%% SLIP_DEMO a demo of SLIP_backslash
% SLIP_LU is a package for solving sparse linear systems of equations
% with a roundoff-free integer-preserving method.  The result is
% always exact, unless the matrix A is perfectly singular.
%
% See also vpa, SLIP_backslash, SLIP_install, SLIP_test.
%
% SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
% Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
% SLIP_LU/License for the license.

format compact

%% SLIP_backslash vs MATLAB backslash: first example
% In this first example, x = SLIP_backslash (A,b) returns an approximate
% solution, but not because it was computed incorrectly in SLIP_backslash.
% It is computed exactly as a rational result in SLIP_backslash with
% arbitrary precision, but then converted to double precision on output.

format long g
load west0479
A = west0479 ;
n = size (A, 1) ;
xtrue = rand (n,1) ;
b = A*xtrue ;
x = SLIP_backslash (A, b) ;
% error is nonzero: x is computed exactly in rational arbitrary-precision,
% but then lost precision when returned to MATLAB:
err_slip = norm (x-xtrue)
x = A\b ;
% error is nonzero: MATLAB x=A\b experiences floating-point error
% throughout its computations:
err_matlab = norm (x-xtrue)

%% SLIP_backslash: exact, vs MATLAB backslash: approximate
% In this example, x = SLIP_backslash (A,b) is returned exactly in the
% MATLAB vector x, because x contains only integers representable exactly
% in double precision.  x = A\b results in floating-point roundoff error.

amax = max (abs (A), [ ], 'all') ;
A = floor (2^20 * (A / amax)) + n * speye (n) ;
xtrue = floor (64 * xtrue) ;
b = A*xtrue ;
x = SLIP_backslash (A, b) ;
% error will be exactly zero:
err_slip = norm (x-xtrue)
x = A\b ;
% error will be small but nonzero:
err_matlab = norm (x-xtrue)

%% SLIP_backslash on ill-conditioned problems
% x = SLIP_backslash (A,b) is able to solve problems that x=A\b cannot.
% Consider the following matrix in the MATLAB gallery:

[U, b] = gallery ('wilk', 3)

%%    vpa can find a good but not perfect solution:
xvpa = vpa (U) \ b

%     but MATLAB's numerical x = U\b computes a poor solution:
xapprox = U \ b

%% SLIP_backslash computes the exact answer
% It returns it to MATLAB as a double vector, obtaining the exact results,
% except for a final floating-point error in x(2):

xslip = SLIP_backslash (U, b)
err = xvpa - xslip
relerr = double (err (2:3) ./ xvpa (2:3))

%% SLIP_backslash with exact results
% SLIP_backslash can also return x as a cell array of strings, which
% preserves the exact rational result.  The printing option is also
% enabled in this example.  The floating-point matrices U and b are
% converted into a scaled integer matrix before solving U*x=b with
% SLIP LU.
%
% The value U(1,2)=0.9 is a floating-point number, and 0.9 cannot be
% exactly represented in IEEE floating-point representation.  It is
% converted exactly into the rational number,
% fl(0.9) = 45000000000000001 / 50000000000000000.

option.print = 3 ;          % also print the details
option.solution = 'char' ;  % return x as a cell array of strings

%%

xslip = SLIP_backslash (U, b, option)

%% Converting an exact rational result to vpa or double
% If SLIP_backslash returns x as a cell array of strings, it cannot
% be immediately used in computations in MATLAB.  It can be converted
% into a vpa or double matrix, as illustrated below.  The solution
% differs slightly from the vpa solution xvpa = vpa (U)\b, since
% the MATLAB vpa converts fl(0.9) into a decimal representation 0.9,
% or exactly 9/10; this is not exactly equal to fl(0.9), since the
% value 9/10 is not representable in IEEE floating-point.  SLIP_backslash,
% by contrast, converts fl(0.9) into its exact rational representation,
% 45000000000000001 / 50000000000000000.

xslip_as_vpa = vpa (xslip)
xslip_as_double = double (vpa (xslip))
xvpa_as_double = double (xvpa)

%% Comparing the VPA and SLIP_BACKSLASH solutions in double
% Both vpa(U)\b and SLIP_backslash(U,b) compute the same result
% in the end, when their results are converted to double.
err = xvpa_as_double - xslip_as_double

