function x = mldivide (F,b,is_inverse)
%MLDIVIDE x = A\b using the factorization F = factorize(A)
%
% Example
%   F = factorize(A) ;
%   x = F\b ;               % same as x = A\b
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

if (nargin < 3)
    is_inverse = F.is_inverse ;
end

if (is_inverse)

    % x=inverse(A)\b is a double inverse, so it becomes simply x=A*b
    x = F.A*b ;

else

    kind = F.kind ;
    switch kind

        case 1

            % least-squares solution of an overdetermined problem
            % Q-less econonmy sparse QR factorization: (A*q)'*(A*q) = R'*R
            A = F.A ;
            R = F.R ;
            q = F.q ;
            x = q * (R \ (R' \ (q' * (A' * b)))) ;
            e = q * (R \ (R' \ (q' * (A' * (b - A * x))))) ;
            x = x + e ;

        case 2

            % least-squares solution of an overdetermined problem
            % dense economy QR factorization: A = Q*R
            % x = R \ (Q' * b)
            Q = F.Q ;
            R = F.R ;
            if (issparse (b))
                b = full (b) ;
            end
            opU.UT = true ;
            x = linsolve (R, Q' * b, opU) ;

        case 3

            % minimum 2-norm solution of an underdetermined system
            % Q-less economy sparse QR factorization: (p*A)*(p*A)' = R'*R
            A = F.A ;
            R = F.R ;
            p = F.p ;
            x = A' * (p' * (R \ (R' \ (p * b)))) ;
            e = A' * (p' * (R \ (R' \ (p * (b - A * x))))) ;
            x = x + e ;

        case 4

            % minimum 2-norm solution of an underdetermined system
            % dense economy QR factorization: A' = Q*R
            Q = F.Q ;
            R = F.R ;
            if (issparse (b))
                b = full (b) ;
            end
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = Q * linsolve (R, b, opUT) ;

        case 5

            % sparse Cholesky factorization: q*A*q' = L*L'
            L = F.L ;
            q = F.q ;
            x = q * (L' \ (L \ (q' * b))) ;

        case 6

            % dense Cholesky factorization: A = R'*R
            % x = R \ (R' \ b)
            R = F.R ;
            if (issparse (b))
                b = full (b) ;
            end
            opU.UT = true ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = linsolve (R, linsolve (R, b, opUT), opU) ;

        case 7

            % sparse LU factorization: p*A*q = L*U
            L = F.L ;
            U = F.U ;
            p = F.p ;
            q = F.q ;
            x = q * (U \ (L \ (p * b))) ;

        case 8

            % dense LU factorization: p*A = L*U
            % x = U \ (L \ (p*b)) ;
            L = F.L ;
            U = F.U ;
            p = F.p ;
            if (issparse (b))
                b = full (b) ;
            end
            opL.LT = true ;
            opU.UT = true ;
            x = linsolve (U, linsolve (L, p*b, opL), opU) ;

    end
end

