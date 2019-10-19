classdef factorization_sparse_qr < factorization_generic
%FACTORIZATION_SPARSE_QR economy sparse QR of A: (A*q)'*(A*q) = R'*R

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_sparse_qr (A)
            %FACTORIZATION_SPARSE_QR economy sparse QR: (A*q)'*(A*q) = R'*R
            n = size (A,2) ;
            q = sparse (colamd (A), 1:n, 1) ;
            R = qr (A*q, 0) ;
            assert (nnz (diag (R)) == n, 'Matrix is rank deficient.') ;
            F.A = A ;
            F.R = R ;
            F.q = q ;
        end

        function disp (F)
            %DISP displays a Q-less economy sparse QR
            fprintf ('  Q-less economy sparse QR factorization of A: ') ;
            fprintf ('(A*q)''*A*q = R''*R\n');
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  R:\n') ; disp (F.R) ;
            fprintf ('  q:\n') ; disp (F.q) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBLCASS x = A\b using economy sparse QR of A
            % least-squares solution of an overdetermined problem
            A = F.A ;
            R = F.R ;
            q = F.q ;
            x = q * (R \ (R' \ (q' * (A' * b)))) ;
            e = q * (R \ (R' \ (q' * (A' * (b - A * x))))) ;
            x = x + e ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using economy sparse QR of A
            % minimum 2-norm solution of an underdetermined problem
            bT = b' ;
            A = F.A ;
            R = F.R ;
            q = F.q ;
            x = A * (q * (R \ (R' \ (q' * bT)))) ;
            e = A * (q * (R \ (R' \ (q' * (bT - A' * x))))) ;
            x = (x + e)' ;
        end
    end
end
