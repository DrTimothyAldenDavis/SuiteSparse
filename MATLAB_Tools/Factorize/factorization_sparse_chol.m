classdef factorization_sparse_chol < factorization_generic
%FACTORIZATION_SPARSE_CHOL sparse Cholesky factorization: q*A*q' = L*L'

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_sparse_chol (A)
            %FACTORIZATION_SPARSE_CHOL sparse Cholesky: q*A*q' = L*L'
            [L g q] = chol (A, 'lower') ;
            assert (g == 0, 'Matrix is not positive definite.') ;
            F.A = A ;
            F.L = L ;
            F.q = q ;
        end

        function disp (F)
            %DISP displays a sparse Cholesky factorization
            fprintf ('  sparse Cholesky factorization: q*A*q'' = L*L''\n');
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  L:\n') ; disp (F.L) ;
            fprintf ('  q:\n') ; disp (F.q) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using sparse Cholesky
            % x = q * (L' \ (L \ (q' * b)))
            L = F.L ;
            q = F.q ;
            x = q * (L' \ (L \ (q' * b))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse Cholesky
            % x = (q * (L' \ (L \ (q' * b'))))'
            x = (mldivide_subclass (F,b'))' ;
        end
    end
end
