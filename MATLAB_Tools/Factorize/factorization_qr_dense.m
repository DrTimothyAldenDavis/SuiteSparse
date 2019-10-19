classdef factorization_qr_dense < factorization
%FACTORIZATION_QR_DENSE A = Q*R where A is full.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_qr_dense (A, fail_if_singular)
            %FACTORIZATION_QR_DENSE : A = Q*R
            [m n] = size (A) ;
            if (m < n)
                error ('FACTORIZE:wrongdim', 'QR(A) method requires m >= n.') ;
            end
            [f.Q f.R] = qr (A,0) ;
            F.A_condest = cheap_condest (get_diag (f.R), fail_if_singular) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.kind = 'dense economy QR factorization: A = Q*R' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBLCASS x = A\b using a dense economy QR factorization
            % least-squares solution of an overdetermined problem
            % x = R \ (Q' * b)
            f = F.Factors ;
            opU.UT = true ;
            x = linsolve (f.R, f.Q' * full (b), opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense QR of A
            % minimum 2-norm solution of a underdetermined problem
            % x = (Q * (R' \ b'))' ;
            f = F.Factors ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = (f.Q * linsolve (f.R, full (b'), opUT))' ;
        end
    end
end
