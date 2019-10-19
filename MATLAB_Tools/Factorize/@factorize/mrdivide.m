function x = mrdivide (b,F,is_inverse)
%MRDIVIDE x = b/A using the factorization F = factorize(A)
%
% Example
%   F = factorize(A) ;
%   x = b/F ;               % same as x=b/A
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

if (nargin < 3)
    is_inverse = F.is_inverse ;
end

if (is_inverse)

    % x=b/inverse(A) is a double inverse, so it becomes simply x=b*A
    x = b*F.A ;

else

    bT = b' ;
    kind = F.kind ;
    switch kind

        case 1

            % minimum 2-norm solution of a sparse underdetermined problem
            % Q-less econonmy sparse QR factorization: (A*q)'*(A*q) = R'*R
            A = F.A ;
            R = F.R ;
            q = F.q ;
            x = A * (q * (R \ (R' \ (q' * bT)))) ;
            e = A * (q * (R \ (R' \ (q' * (bT - A' * x))))) ;
            x = (x + e)' ;

        case 2

            % minimum 2-norm solution of a dense underdetermined problem
            % dense economy QR factorization: A = Q*R
            % x = (Q * (R' \ b'))' ;
            Q = F.Q ;
            R = F.R ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = (Q * linsolve (R, bT, opUT))' ;

        case 3

            % least-squares solution of a sparse overdetermined problem
            % Q-less economy sparse QR factorization: (p*A)*(p*A)' = R'*R
            A = F.A ;
            R = F.R ;
            p = F.p ;
            x = p' * (R \ (R' \ (p * (A * bT)))) ;
            e = p' * (R \ (R' \ (p * (A * (bT - A' * x))))) ;
            x = (x + e)' ;

        case 4

            % least-squares solution of a dense overdetermined problem
            % dense economy QR factorization: A' = Q*R
            % x = (R \ (Q' * b'))'
            Q = F.Q ;
            R = F.R ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opU.UT = true ;
            x = linsolve (R, Q' * bT, opU)' ;

        case 5

            % sparse Cholesky factorization: q*A*q' = L*L'
            L = F.L ;
            q = F.q ;
            x = (q * (L' \ (L \ (q' * bT))))' ;

        case 6

            % dense Cholesky factorization: A = R'*R
            % x = (R \ (R' \ b'))'
            R = F.R ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opU.UT = true ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = linsolve (R, linsolve (R, bT, opUT), opU)' ;

        case 7

            % sparse LU factorization: p*A*q = L*U
            L = F.L ;
            U = F.U ;
            p = F.p ;
            q = F.q ;
            x = (p' * (L' \ (U' \ (q' * bT))))' ;

        case 8

            % dense LU factorization: p*A = L*U
            % x = (P' * (L' \ (U' \ b')))'
            L = F.L ;
            U = F.U ;
            p = F.p ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opUT.UT = true ;
            opUT.TRANSA = true ;
            opLT.LT = true ;
            opLT.TRANSA = true ;
            x = (p' * linsolve (L, linsolve (U, bT, opUT), opLT))' ;

    end
end

