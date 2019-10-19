function C = subsref (F,ij)
%SUBSREF A(i,j) or (i,j)th entry of inv(A) if F is inverted.
% Otherwise, explicit entries in the inverse are computed.  This method
% also extracts the contents of F (A, L, U, Q, R, p, q, is_inverse, and
% kind).
%
% Example
%   F = factorize(A)
%   F(1,2)              % same as A(1,2)
%   F.L                 % the L factor of the factorization of A
%   S = inverse(A)
%   S(1,2)              % the (1,2) entry of inv(A), but only computes
%                       % the 2nd column of inv(A) via backslash.
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

switch (ij(1).type)

    case '.'

        % F.U usage, for example: extract one of the matrices from F
        if (length (ij) > 2)
            error ('Improper index matrix reference.') ;
        end

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
            case 'kind'
                C = F.kind ;
            otherwise
                error ('Reference to non-existent field ''%s''.', ...
                    ij(1).subs) ;
        end

        % F.U(2,3) usage, return U(2,3)
        if (length (ij) > 1)
            C = subsref (C, ij (2)) ;
        end

    case '()'

        % F(2,3) usage, return A(2,3) or the (2,3) entry of inv(A).
        if (length (ij) > 1)
            error ('Improper index matrix reference.') ;
        end
        A = F.A ;
        if (F.is_inverse)
            % The caller is requesting explicit entries of the inverse.
            if (length (ij.subs) ~= 2)
                error ('Linear indexing of inverse not supported.') ;
            end
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
                % For F(i,j), compute cols S(:,j) of the inverse S=inv(A)
                if (issparse (A))
                    I = speye (m) ;
                else
                    I = eye (m) ;
                end
                C = subsref (mldivide (F, subsref (I,j), 0), i) ;
            else
                % For F(i,j), compute rows S(i,:) of the inverse S=inv(A)
                if (issparse (A))
                    I = speye (n) ;
                else
                    I = eye (n) ;
                end
                C = subsref (mrdivide (subsref (I,i), F, 0), j) ;
            end
        else
            % F is not inverted, so just return A(i,j)
            C = subsref (A, ij) ;
        end

    case '{}'
        
        error ('Cell contents reference from a non-cell array object.') ;
end

