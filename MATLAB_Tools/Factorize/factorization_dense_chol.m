classdef factorization_dense_chol < factorization_generic
%FACTORIZE_DENSE_CHOL a dense Cholesky factorization, A = R'*R

% Copyright 2009, Timothy A. Davis, University of Florida

    methods

        function F = factorization_dense_chol (A)
            %FACTORIZATION_DENSE_CHOL dense Cholesky: A = R'*R
            [R g] = chol (A) ;
            assert (g == 0, 'Matrix is not positive definite.') ;
            F.A = A ;
            F.R = R ;
        end

        function disp (F)
            %DISP displays a dense Cholesky factorization
            fprintf ('  dense Cholesky factorization: A = R''*R\n') ;
            fprintf ('  A:\n') ; disp (F.A) ;
            fprintf ('  R:\n') ; disp (F.R) ;
            fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        end

        function x = mldivide_subclass (F,b)
            %MLDIVIDE_SUBCLASS x=A\b using a dense Cholesky factorization
            % x = R \ (R' \ b)
            R = F.R ;
            if (issparse (b))
                b = full (b) ;
            end
            opU.UT = true ;
            opUT.UT = true ;
            opUT.TRANSA = true ;
            x = linsolve (R, linsolve (R, b, opUT), opU) ;
        end

        function x = mrdivide_subclass (b,F)
            %MRDIVIDE_SUBCLASS x = b/A using dense Cholesky
            % x = (R \ (R' \ b')'
            x = (mldivide_subclass (F, b'))' ;
        end

        function F = plus (F,w)
            %PLUS update a dense Cholesky factorization
            %
            % Example
            %   % F becomes the Cholesky factorization of A+w*w'
            %   F = factorize (A) ;
            %   w = rand (size (A,1),1) ;
            %   F = F + w ;
            %   x = F\b ;               % computes x = (A+w*w')\b
            %
            % See also factorize, cholupdate.
            F.R = cholupdate (F.R, w, '+') ;
            F.A = F.A + w*w' ;
        end

        function F = minus (F,w)
            %MINUS downdate a dense Cholesky factorization
            %
            % Example
            %   % F becomes the Cholesky factorization of A-w*w'
            %   F = factorize (A) ;
            %   w = rand (size (A,1),1) ;
            %   F = F - w ;
            %   x = F\b ;               % computes x = (A-w*w')\b
            %
            % See also factorize, cholupdate.
            F.R = cholupdate (F.R, w, '-') ;
            F.A = F.A - w*w' ;
        end
    end
end
