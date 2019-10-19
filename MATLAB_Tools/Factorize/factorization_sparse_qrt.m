classdef factorization_sparse_qrt < factorization_generic
%FACTORIZATION_SPARSE_QRT economy sparse QR of A': (p*A)*(p*A)'=R'*R

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_sparse_qrt (A)
            %FACTORIZATION_SPARSE_QRT economy sparse QR: (p*A)*(p*A)'=R'*R
            m = size (A,1) ;
            C = A' ;
            p = sparse (1:m, colamd (C), 1) ;
            R = qr (C*p', 0) ;
            assert (nnz (diag (R)) == m, 'Matrix is rank deficient.') ;
            F.A = A ;
            F.R = R ;
            F.p = p ;
        end

        function disp (F)
            %DISP displays a Q-less economy sparse QR
            fprintf ('  Q-less economy sparse QR factorization of A'': ') ;
            fprintf ('(p*A)*(p*A)'' = R''*R\n') ;
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  R:\n') ; disp (F.R) ;
            fprintf ('  p:\n') ; disp (F.p) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end


        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using economy sparse QR of A'
            % minimum 2-norm solution of an underdetermined system
            A = F.A ;
            R = F.R ;
            p = F.p ;
            x = A' * (p' * (R \ (R' \ (p * b)))) ;
            e = A' * (p' * (R \ (R' \ (p * (b - A * x))))) ;
            x = x + e ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using economy sparse QR of A'
            % least-squares solution of an overdetermined problem
            bT = b' ;
            A = F.A ;
            R = F.R ;
            p = F.p ;
            x = p' * (R \ (R' \ (p * (A * bT)))) ;
            e = p' * (R \ (R' \ (p * (A * (bT - A' * x))))) ;
            x = (x + e)' ;
        end
    end
end
