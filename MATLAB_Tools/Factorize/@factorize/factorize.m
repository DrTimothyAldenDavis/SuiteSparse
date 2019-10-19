classdef factorize
%FACTORIZE an object-oriented method for solving linear systems
% and least-squares problems, and for representing operations with the
% inverse of a square matrix or the pseudo-inverse of a rectangular matrix.
%
% F = factorize(A) returns an object F that holds the factorization of a
% non-singular matrix A.  x=F\b then solves a linear system or a
% least-squares problem.  S=inverse(F) or S=inverse(A) returns a factorized
% representation of the inverse of A so that inverse(A)*b is mathematically
% equivalent to inv(A)*b, but the former does not actually compute the
% inverse of A.
%
% Example
%
%   F = factorize(A) ;      % LU, QR, or Cholesky factorization of A
%   x = F\b ;               % solve A*x=b; same as x=A\b
%   S = inverse (F) ;       % S represents the factorization of inv(A)
%   x = S*b ;               % same as x = A\b.
%   S = A-B*inverse(D)*C    % efficiently computes the Schur complement
%   S = A-B*inv(D)*C        % bad method for computing the Schur complement
%   S = inverse(A) ; S(:,1) % compute just the first column of inv(A),
%                           % without computing inv(A)
%
% If A is square, symmetric (Hermitian for the complex case), and has a
% real positive diagonal, then use F=factorize(A,1).  If you know
% otherwise, use F=factorize(A,0).  Using this option improves performance,
% since otherwise this condition must be checked to choose between a
% Cholesky or LU factorization.  The option is ignored if A is rectangular.
%
% For more details, type "help factorize1".
% For a demo type "fdemo" or see the html/ directory.
%
% See also inverse, factorize1, mldivide, mrdivide, inv, pinv, linsolve

% Copyright 2009, Timothy A. Davis, University of Florida

    properties (SetAccess = protected)
        % The factorize object holds a QR, LU, Cholesky factorization:
        A = [ ] ;           % a copy of the input matrix
        L = [ ] ;           % lower-triangular factor for LU and Cholesky
        U = [ ] ;           % upper-triangular factor for LU
        Q = [ ] ;           % Q factor for dense QR
        R = [ ] ;           % R factor for QR
        p = [ ] ;           % sparse row permutation matrix
        q = [ ] ;           % sparse column permutation matrix
        is_inverse = false ;% F represents the factorization of A or inv(A)
        kind = 0 ;          % F is one of 8 kinds of factorizations
    end

    methods

        function F = factorize (A,try_chol)

            % factorize constructor: compute a factorization of A

            if (ndims (A) > 2)
                error ('Matrix must be 2D.') ;
            end
            [m n] = size (A) ;
            F.A = A ;

            if (m > n)

                % QR factorization of A
                if (issparse (A))
                    % Q-less econonmy sparse QR: (A*q)'*(A*q) = R'*R
                    q = sparse (colamd (A), 1:n, 1) ;
                    R = qr (A*q, 0) ;
                    F.q = q ;
                    F.kind = 1 ;
                else
                    % dense economy QR factorization: A = Q*R
                    [Q R] = qr (A,0) ;
                    F.Q = Q ;
                    F.kind = 2 ;
                end
                ok = (nnz (diag (R)) == n) ;
                F.R = R ;

            elseif (m < n)

                % QR factorization of A'
                if (issparse (A))
                    % Q-less economy sparse QR: (p*A)*(p*A)' = R'*R
                    C = A' ;
                    p = sparse (1:m, colamd (C), 1) ;
                    R = qr (C*p', 0) ;
                    F.p = p ;
                    F.kind = 3 ;
                else
                    % dense economy LQ factorization: A' = Q*R
                    [Q R] = qr (A',0) ;
                    F.Q = Q ;
                    F.kind = 4 ;
                end
                ok = (nnz (diag (R)) == m) ;
                F.R = R ;

            else

                % Cholesky or LU factorization of A
                g = 1 ;
                if (nargin == 1)
                    % This is an expensive test, so skip it if the caller
                    % already knows the matrix is a candidate for Cholesky.
                    d = diag (A) ;
                    try_chol = (all (d > 0) && nnz (imag (d)) == 0 && ...
                        nnz (A-A') == 0) ;
                end
                if (try_chol)
                    if (issparse (A))
                        % sparse Cholesky factorization: q'*A*q = L*L'
                        [L g q] = chol (A, 'lower') ;
                    else
                        % dense Cholesky factorization: A = R'*R
                        [R g] = chol (A) ;
                    end
                end
                % do an LU factorization if Cholesky failed or was skipped
                ok = (g == 0) ;
                if (ok)
                    % Cholesky was successful
                    if (issparse (A))
                        F.L = L ;
                        F.q = q ;
                        F.kind = 5 ;
                    else
                        F.R = R ;
                        F.kind = 6 ;
                    end
                else
                    % need an LU factorization
                    if (issparse (A))
                        % sparse LU factorization: p*A*q = L*U
                        [L U p q] = lu (A) ;
                        F.q = q ;
                        F.kind = 7 ;
                    else
                        % dense LU factorization: p*A = L*U
                        [L U p] = lu (A, 'vector') ;
                        p = sparse (1:n, p, 1) ;
                        F.kind = 8 ;
                    end
                    F.L = L ;
                    F.U = U ;
                    F.p = p ;
                    ok = (nnz (diag (U)) == n)  ;
                end
            end

            if (~ok)
                error ('Matrix is rank deficient.') ;
            end
        end
    end
end

