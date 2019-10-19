classdef factorization_sparse_lu < factorization_generic
%FACTORIZATION_SPARSE_LU sparse LU factorization: p*A*q = L*U

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_sparse_lu (A)
            %FACTORIZATION_SPARSE_LU sparse LU: p*A*q = L*U
            n = size (A,2) ;
            [F.L U F.p F.q] = lu (A) ;
            assert (nnz (diag (U)) == n, 'Matrix is rank deficient.') ;
            F.U = U ;
            F.A = A ;
        end

        function disp (F)
            %DISP displays a sparse LU factorization
            fprintf ('  sparse LU factorization: p*A*q = L*U\n') ;
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  L:\n') ; disp (F.L) ;
            fprintf ('  U:\n') ; disp (F.U) ;
            fprintf ('  p:\n') ; disp (F.p) ;
            fprintf ('  q:\n') ; disp (F.q) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using sparse LU
            % x = q * (U \ (L \ (p * b)))
            x = F.q * (F.U \ (F.L \ (F.p * b))) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using sparse LU
            % x = (p' * (L' \ (U' \ (q' * b'))))'
            x = (F.p' * (F.L' \ (F.U' \ (F.q' * b'))))' ;
        end

    end
end
