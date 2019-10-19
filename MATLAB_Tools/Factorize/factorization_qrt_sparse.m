classdef factorization_qrt_sparse < factorization
%FACTORIZATION_QRT_SPARSE (P*A)*(P*A)'=R'*R where A is sparse.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_qrt_sparse (A, fail_if_singular)
            %FACTORIZATION_QRT_SPARSE economy sparse QR: (P*A)*(P*A)'=R'*R
            if (~isa (A, 'double'))
                error ('FACTORIZE:wrongtype', 'A must be double') ;
            end
            [m n] = size (A) ;
            if (m >= n)
                error ('FACTORIZE:wrongdim', 'QR of A'' requires m < n.') ;
            end
            C = A' ;
            f.P = sparse (1:m, colamd (C), 1) ;
            f.R = qr (C*f.P', 0) ;
            F.A_condest = cheap_condest (get_diag (f.R), fail_if_singular) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = m ;
            F.kind = 'sparse QR factorization of A'': (P*A)*(P*A)'' = R''*R' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using economy sparse QR of A'
            % minimum 2-norm solution of an underdetermined system
            A = F.A ;
            f = F.Factors ;
            x = A' * (f.P' * (f.R \ (f.R' \ (f.P * b)))) ;
            e = A' * (f.P' * (f.R \ (f.R' \ (f.P * (b - A * x))))) ;
            x = x + e ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using economy sparse QR of A'
            % least-squares solution of an overdetermined problem
            bT = b' ;
            A = F.A ;
            f = F.Factors ;
            x = f.P' * (f.R \ (f.R' \ (f.P * (A * bT)))) ;
            e = f.P' * (f.R \ (f.R' \ (f.P * (A * (bT - A' * x))))) ;
            x = (x + e)' ;
        end
    end
end
