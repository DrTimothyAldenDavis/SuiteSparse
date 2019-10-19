classdef factorization_qr_sparse < factorization
%FACTORIZATION_QR_SPARSE (A*P)'*(A*P) = R'*R where A is sparse.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_qr_sparse (A, fail_if_singular)
            %FACTORIZATION_QR_SPARSE economy sparse QR: (A*P)'*(A*P) = R'*R
            if (~isa (A, 'double'))
                error ('FACTORIZE:wrongtype', 'A must be double') ;
            end
            [m n] = size (A) ;
            if (m < n)
                error ('FACTORIZE:wrongdim', 'QR of A requires m >= n.') ;
            end
            f.P = sparse (colamd (A), 1:n, 1) ;
            f.R = qr (A*f.P, 0) ;
            F.A_condest = cheap_condest (get_diag (f.R), fail_if_singular) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.kind = 'sparse QR factorization of A: (A*P)''*A*P = R''*R' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBLCASS x = A\b using economy sparse QR of A
            % least-squares solution of an overdetermined problem
            A = F.A ;
            f = F.Factors ;
            x = f.P * (f.R \ (f.R' \ (f.P' * (A' * b)))) ;
            e = f.P * (f.R \ (f.R' \ (f.P' * (A' * (b - A * x))))) ;
            x = x + e ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using economy sparse QR of A
            % minimum 2-norm solution of an underdetermined problem
            bT = b' ;
            A = F.A ;
            f = F.Factors ;
            x = A * (f.P * (f.R \ (f.R' \ (f.P' * bT)))) ;
            e = A * (f.P * (f.R \ (f.R' \ (f.P' * (bT - A' * x))))) ;
            x = (x + e)' ;
        end
    end
end
