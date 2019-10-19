function F = factorize (A,try_chol)
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
% See also factorization_generic, inverse, factorize1, mldivide,
% mrdivide, inv, pinv, linsolve

% Copyright 2009, Timothy A. Davis, University of Florida

assert (ndims (A) == 2, 'Matrix must be 2D.') ;
[m n] = size (A) ;

if (m > n)

    % QR factorization of A
    if (issparse (A))
        % Q-less econonmy sparse QR: (A*q)'*(A*q) = R'*R
        F = factorization_sparse_qr (A) ;
    else
        % dense economy QR factorization: A = Q*R
        F = factorization_dense_qr (A) ;
    end

elseif (m < n)

    % QR factorization of A'
    if (issparse (A))
        F = factorization_sparse_qrt (A) ;
    else
        % dense economy LQ factorization: A' = Q*R
        F = factorization_dense_qrt (A) ;
    end

else

    % Cholesky or LU factorization of A
    chol_ok = false ;
    if (nargin == 1)
        % This is an expensive test, so skip it if the caller
        % already knows the matrix is a candidate for Cholesky.
        d = diag (A) ;
        try_chol = (all (d > 0) && nnz (imag (d)) == 0 && nnz (A-A') == 0);
    end
    if (try_chol)
        try
            if (issparse (A))
                % sparse Cholesky factorization: q'*A*q = L*L'
                F = factorization_sparse_chol (A) ;
            else
                % dense Cholesky factorization: A = R'*R
                F = factorization_dense_chol (A) ;
            end
            chol_ok = true ;
        catch %#ok
            % Cholesky failed
        end
    end

    % do an LU factorization if Cholesky failed or was skipped
    if (~chol_ok)
        % need an LU factorization
        if (issparse (A))
            F = factorization_sparse_lu (A) ;
        else
            F = factorization_dense_lu (A) ;
        end
    end
end
