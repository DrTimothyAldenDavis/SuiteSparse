function disp (F)
%DISP displays the factorization F
%
% Example
%   F = factorize (A)
%   disp (F)
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

fprintf ('  A:\n') ;
disp (F.A) ;

if (~isempty (F.L))
    fprintf ('  L:\n') ;
    disp (F.L) ;
end

if (~isempty (F.U))
    fprintf ('  U:\n') ;
    disp (F.U) ;
end

if (~isempty (F.Q))
    fprintf ('  Q:\n') ;
    disp (F.Q) ;
end

if (~isempty (F.R))
    fprintf ('  R:\n') ;
    disp (F.R) ;
end

if (~isempty (F.p))
    fprintf ('  p:\n') ;
    disp (F.p) ;
end

if (~isempty (F.q))
    fprintf ('  q:\n') ;
    disp (F.q) ;
end

fprintf ('  is_inverse: %d kind: %d\n', F.is_inverse, F.kind) ;

% print the kind of factorization that F contains

switch F.kind

    case 1

        fprintf ('  Q-less economy sparse QR factorization: ') ;
        fprintf ('(A*q)''*A*q = R''*R\n');

    case 2

        fprintf ('  dense economy QR factorization: A = Q*R\n') ;

    case 3

        fprintf ('  Q-less economy sparse QR factorization: ') ;
        fprintf ('(p*A)*(p*A)'' = R''*R\n') ;

    case 4

        fprintf ('  dense economy QR factorization: A'' = Q*R\n') ;

    case 5

        fprintf ('  sparse Cholesky factorization: ')  ;
        fprintf ('q*A*q'' = L*L''\n');

    case 6

        fprintf ('  dense Cholesky factorization: A = L*L''\n') ;

    case 7

        fprintf ('  sparse LU factorization: p*A*q = L*U\n') ;

    case 8

        fprintf ('  dense LU factorization: p*A = L*U\n') ;

end

