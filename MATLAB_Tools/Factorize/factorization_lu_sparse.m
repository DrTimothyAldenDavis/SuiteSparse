classdef factorization_lu_sparse < factorization
%FACTORIZATION_LU_SPARSE P*A*Q = L*U where A is square and sparse.

% Copyright 2011-2012, Timothy A. Davis, http://www.suitesparse.com

    methods

        function F = factorization_lu_sparse (A, fail_if_singular)
            %FACTORIZATION_LU_SPARSE : P*A*Q = L*U
            [m n] = size (A) ;
            if (m ~= n)
                error ('FACTORIZE:wrongdim', ...
                    'LU for rectangular matrices not supported.  Use QR.') ;
            end
            [f.L f.U f.P f.Q] = lu (A) ;
            F.A_condest = cheap_condest (get_diag (f.U), fail_if_singular) ;
            F.A = A ;
            F.Factors = f ;
            F.A_rank = n ;
            F.kind = 'sparse LU factorization: P*A*Q = L*U' ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using sparse LU
            % x = Q * (U \ (L \ (P * b)))
            f = F.Factors ;
            x = f.Q * (f.U \ (f.L \ (f.P * b))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse LU
            % x = (P' * (L' \ (U' \ (Q' * b'))))'
            f = F.Factors ;
            x = (f.P' * (f.L' \ (f.U' \ (f.Q' * b'))))' ;
        end
    end
end
