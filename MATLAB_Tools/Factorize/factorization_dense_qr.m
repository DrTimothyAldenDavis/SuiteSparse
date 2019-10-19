classdef factorization_dense_qr < factorization_generic
%FACTORIZATION_DENSE_QR dense economy QR factorization of A: A = Q*R

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_dense_qr (A)
            %FACTORIZATION_DENSE_QR dense economy QR: A = Q*R
            n = size (A,2) ;
            [F.Q R] = qr (A,0) ;
            assert (nnz (diag (R)) == n, 'Matrix is rank deficient.') ;
            F.R = R ;
            F.A = A ;
        end

        function disp (F)
            %DISP displays a dense economy QR factorization
            fprintf ('  dense economy QR factorization: A = Q*R\n') ;
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  Q:\n') ; disp (F.Q) ;
            fprintf ('  R:\n') ; disp (F.R) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE x = A\b using a dense economy QR factorization
            % least-squares solution of an overdetermined problem
            % x = R \ (Q' * b)
            if (issparse (b))
                b = full (b) ;
            end
            opU.UT = true ;
            x = linsolve (F.R, F.Q' * b, opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense QR of A
            % minimum 2-norm solution of a underdetermined problem
            % x = (Q * (R' \ b'))' ;
            bT = b' ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = (F.Q * linsolve (F.R, bT, opUT))' ;
        end
    end
end
