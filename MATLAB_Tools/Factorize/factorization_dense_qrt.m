classdef factorization_dense_qrt < factorization_generic
%FACTORIZATION_DENSE_QRT dense economy QR of A': A' = Q*R

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_dense_qrt (A)
            %FACTORIZATION_DENSE_QRT dense economy QR: A' = Q*R
            m = size (A,1) ;
            [F.Q R] = qr (A',0) ;
            assert (nnz (diag (R)) == m, 'Matrix is rank deficient.') ;
            F.R = R ;
            F.A = A ;
        end

        function disp (F)
            %DISP displays a dense economy QR factorization: A' = Q*R
            fprintf ('  dense economy QR factorization: A'' = Q*R\n') ;
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  Q:\n') ; disp (F.Q) ;
            fprintf ('  R:\n') ; disp (F.R) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x = A\b using a dense QR factorization of A'
            % minimum 2-norm solution of an underdetermined system
            % x = Q * (R' \ b)
            if (issparse (b))
                b = full (b) ;
            end
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = F.Q * linsolve (F.R, b, opUT) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense QR of A'
            % least-squares solution of a dense overdetermined problem
            % x = (R \ (Q' * b'))'
            bT = b' ;
            if (issparse (bT))
                bT = full (bT) ;
            end
            opU.UT = true ;
            x = linsolve (F.R, F.Q' * bT, opU)' ;
        end
    end
end
