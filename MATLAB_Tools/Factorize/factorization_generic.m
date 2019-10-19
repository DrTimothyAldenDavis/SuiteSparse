classdef factorization_generic
%FACTORIZATION_GENERIC a generic matrix factorization object
%
% This is an abstract class that is specialized into 8 different kinds of
% matrix factorizations:
%
%   factorization_dense_chol    dense Cholesky      A = R'*R
%   factorization_dense_lu      dense LU            p*A = L*U
%   factorization_dense_qr      dense QR of A       A = Q*R
%   factorization_dense_qrt     dense QR of A'      A' = Q*R
%   factorization_sparse_chol   sparse Cholesky     q*A*q' = L*L'
%   factorization_sparse_lu     sparse LU           p*A*q = L*U
%   factorization_sparse_qr     sparse QR of A      (A*q)'*(A*q) = R'*R
%   factorization_sparse_qrt    sparse QR of A'     (p*A)*(p*A)' = R'*R
%
% The abstract class provides these functions:
%
%   disp (F)                abstract, displays the factorization
%   x = mldivide (F,b)      x = A\b using the factorization F
%   x = mrdivide (b,F)      x = A\b using the factorization F
%   F = inverse (F)         flags F as the factorization of inv(A)
%   S = double (F)          returns F as a matrix (A or inv(A))
%   e = end (F,k,n)         for use in subsref
%   [m n] = size (F,k)      size(A)
%   x = mtimes (y,z)        A*b, inv(A)*b, b*A, or b*inv(A)
%   C = subsref (F,ij)      return components of F, A, or inv(A)
%
% The mldivide and mrdivide functions are supported by methods specific
% to each kind factorization (mldivide_subclass and mrdivide_subclass).
%
% The factorization_dense_chol also supports plus and minus (an interface
% to cholupdate).
%
% See also factorization_dense_chol, factorization_dense_lu,
%    factorization_dense_qr, factorization_dense_qrt,
%    factorization_sparse_chol, factorization_sparse_lu,
%    factorization_sparse_qr, factorization_sparse_qrt

% Copyright 2009, Timothy A. Davis, University of Florida

    properties (SetAccess = protected)
        % The abstract class holds a QR, LU, Cholesky factorization:
        A = [ ] ;           % a copy of the input matrix
        L = [ ] ;           % L factor for LU and sparse Cholesky
        U = [ ] ;           % U factor for LU
        Q = [ ] ;           % Q factor for dense QR
        R = [ ] ;           % R factor for QR or dense Cholesky
        p = [ ] ;           % sparse row permutation matrix
        q = [ ] ;           % sparse column permutation matrix
        is_inverse = false ;% F represents the factorization of A or inv(A)
    end

    methods (Abstract)
        disp (F) ;
        x = mldivide_subclass (F, b) ;
        x = mrdivide_subclass (b, F) ;
    end

    methods

        function x = mldivide (F,b,is_inverse)
            %MLDIVIDE x = A\b using the factorization F = factorize(A)
            %
            % Example
            %   F = factorize(A) ;
            %   x = F\b ;               % same as x = A\b
            %
            % See also factorize.
            if (nargin < 3)
                is_inverse = F.is_inverse ;
            end
            if (is_inverse)
                % x=inverse(A)\b is a double inverse, so it becomes x=A*b
                x = F.A*b ;
            else
                % x=A\b using the factorization-specific method
                x = mldivide_subclass (F, b) ;
            end
        end

        function x = mrdivide (b,F,is_inverse)
            %MRDIVIDE x = b/A using the factorization F = factorize(A)
            %
            % Example
            %   F = factorize(A) ;
            %   x = b/F ;               % same as x=b/A
            %
            % See also factorize.
            if (nargin < 3)
                is_inverse = F.is_inverse ;
            end
            if (is_inverse)
                % x=b/inverse(A) is a double inverse, so it becomes x=b*A
                x = b*F.A ;
            else
                % x=b/A using the factorization-specific method
                x = mrdivide_subclass (b, F) ;
            end
        end

        function F = inverse (F)
            %INVERSE "inverts" F by flagging it as factorization of inv(A)
            %
            % Example
            %
            %   F = factorize (A) ;
            %   S = inverse (F) ;
            %
            % See also factorize.
            F.is_inverse = ~(F.is_inverse) ;
        end

        function S = double (F)
            %DOUBLE returns the factorization as a matrix, A or inv(A)
            %
            % Example
            %   F = factorize (A) ;         % factorizes A
            %   C = double(F) ;             % C = A
            %   S = inv (A)                 % the explicit inverse
            %   S = double (inverse (A))    % same as S=inv(A)
            %
            % See also factorize.
            ij.type = '()' ;
            ij.subs = {':',':'} ;
            S = subsref (F, ij) ;   % let factorize.subsref do all the work
        end

        function e = end (F,k,n)
            %END returns index of last item for use in subsref
            %
            % Example
            %   S = inverse (A) ;   % factorized representation of inv(A)
            %   S (:,end)           % the last column of inv(A),
            %                       % or pinv(A) if A is rectangular
            %
            % See also factorize.
            if (n == 1)
                e = numel (F.A) ;   % # of elements, for linear indexing
            else
                e = size (F,k) ;    % # of rows or columns in A or pinv(A)
            end
        end

        function [m n] = size (F,k)
            %SIZE returns the size of the matrix F.A in the factorization F
            %
            % Example
            %   F = factorize (A)
            %   size(F)                 % same as size (A)
            %
            % See also factorize.
            if (F.is_inverse)
                % swap the dimensions to match pinv(A)
                if (nargout > 1)
                    [n m] = size (F.A) ;
                else
                    m = size (F.A) ;
                    m = m ([2 1]) ;
                end
            else
                if (nargout > 1)
                    [m n] = size (F.A) ;
                else
                    m = size (F.A) ;
                end
            end
            if (nargin > 1)
                m = m (k) ;
            end
        end

        function x = mtimes (y,z)
            %MTIMES A*b, inv(A)*b, b*A, or b*inv(A)
            %
            % Example
            %   S = inverse (A) ;   % factorized representation of inv(A)
            %   x = S*b ;           % same as x=A\b.  Does not use inv(A)
            %
            % See also factorize.
            if (isobject (y))
                if (y.is_inverse)
                    x = mldivide (y,z,0) ;  % x = inv(A)*b via x = A\b
                else
                    x = y.A * z ;           % x = A*b
                end
            else
                if (z.is_inverse)
                    x = mrdivide (y,z,0) ;  % x = b*inv(A) via x = b/A
                else
                    x = y * z.A ;           % x = b*A
                end
            end
        end

        function C = subsref (F,ij)
            %SUBSREF A(i,j) or (i,j)th entry of inv(A) if F is inverted.
            % Otherwise, explicit entries in the inverse are computed.
            % This method also extracts the contents of F (A, L, U, Q, R,
            % p, q, and is_inverse).
            %
            % Example
            %   F = factorize(A)
            %   F(1,2)              % same as A(1,2)
            %   F.L                 % L factor of the factorization of A
            %   S = inverse(A)
            %   S(1,2)              % the (1,2) entry of inv(A), but only
            %                       % computes % the 2nd column of inv(A)
            %                       % via backslash.
            %
            % See also factorize.

            switch (ij(1).type)

                case '.'

                    % F.U usage: extract one of the matrices from F
                    assert (length (ij) <= 2, ...
                        'Improper index matrix reference.') ;

                    switch ij(1).subs
                        case 'A'
                            C = F.A ;
                        case 'L'
                            C = F.L ;
                        case 'U'
                            C = F.U ;
                        case 'Q'
                            C = F.Q ;
                        case 'R'
                            C = F.R ;
                        case 'p'
                            C = F.p ;
                        case 'q'
                            C = F.q ;
                        case 'is_inverse'
                            C = F.is_inverse ;
                        otherwise
                            error (...
                            'Reference to non-existent field ''%s''.', ...
                            ij(1).subs) ;
                    end

                    % F.U(2,3) usage, return U(2,3)
                    if (length (ij) > 1)
                        C = subsref (C, ij (2)) ;
                    end

                case '()'

                    % F(2,3) usage, return A(2,3) or the (2,3) of inv(A).
                    assert (length (ij) == 1, ...
                        'Improper index matrix reference.') ;
                    A = F.A ;
                    if (F.is_inverse)
                        % requesting explicit entries of the inverse
                        assert (length (ij.subs) == 2, ...
                            'Linear indexing of inverse not supported.') ;
                        [m n] = size (A) ;
                        ilen = length (ij.subs {1}) ;
                        if (strcmp (ij.subs {1}, ':'))
                            ilen = n ;
                        end
                        jlen = length (ij.subs {2}) ;
                        if (strcmp (ij.subs {2}, ':'))
                            jlen = m ;
                        end
                        j = ij ;
                        j.subs {1} = ':' ;
                        i = ij ;
                        i.subs {2} = ':' ;
                        if (jlen <= ilen)
                            % compute cols S(:,j) of the inverse S=inv(A)
                            if (issparse (A))
                                I = speye (m) ;
                            else
                                I = eye (m) ;
                            end
                            C = subsref (mldivide (F, subsref (I,j), 0),i);
                        else
                            % compute rows S(i,:) of the inverse S=inv(A)
                            if (issparse (A))
                                I = speye (n) ;
                            else
                                I = eye (n) ;
                            end
                            C = subsref (mrdivide (subsref (I,i), F, 0),j);
                        end
                    else
                        % F is not inverted, so just return A(i,j)
                        C = subsref (A, ij) ;
                    end

                case '{}'
                    
                    error (...
                    'Cell contents reference from a non-cell array object.') ;
            end
        end
    end
end
