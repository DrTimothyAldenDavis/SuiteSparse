classdef factorize1
%FACTORIZE1: a simple and easy-to-read version of factorize.
%
% "Don't let that inv go past your eyes; to solve that system, factorize!"
%
% FACTORIZE1 is a simple object-oriented method for solving linear systems
% of equations that allows for MATLAB expressions that look like they use
% the mathematical inverse, but which use a matrix factorization instead.
% Many mathematical formulas are written as the inverse of a matrix
% multiplied by another matrix.  For example, you might see the Schur
% complement written mathematically as S=A-B*D^(-1)*C.  Don't be tempted to
% write S=A-B*inv(D)*C in MATLAB, which is slow and (worse yet) can be very
% inaccurate.  You should write S=A-B*(D\C) instead, but this is not as
% clear as the A-B*inv(D)*C expression.  With the FACTORIZE object and the
% inverse m-file that uses it, you can write S=A-B*inverse(D)*C, and MATLAB
% will do the right thing for you by computing S=A-B*(D\C) and without
% actually computing the inverse.
%
% You should use the FACTORIZE object for production use, not this one.
% The simple FACTORIZE1 object is meant as an educational tool.  Its "help"
% documentation is long and the code is short, while at the same time
% sacrificing very little performance as compared to FACTORIZE.  FACTORIZE
% is faster and has more features (it can handle rectangular matrices, it
% operates on sparse matrices more efficiently, and it can use Cholesky
% factorization).
%
% Methods in the FACTORIZE1 class:
%
%       F = factorize1(A)   % computes L,U and P so that A=P'*L*U
%       S = inverse(F)      % efficient factorized representation of inv(A)
%
%       % Four equivalent ways to do x=A\b, instead of x=inv(A)*b:
%       x = mldivide(F,b)
%       x = F\b ;
%       x = mtimes(S,b)
%       x = S*b
%
%       % Four equivalent ways to do x=b/A, instead of x=b*inv(A):
%       x = mrdivide(b,F)
%       x = b/F
%       x = mtimes(b,S)
%       x = b*S
%
%       disp(F)             % displays the factors
%   
%       % Since mathematically F=A=P'*L*U, these just multiply A itself:
%       x = mtimes(F,b)     % x=A*b
%       x = F*b             % x=A*b
%       x = mtimes(b,F)     % x=b*A
%       x = b*F             % x=b*A
%
%       C = double(F)       % computes C=inv(A).  "Never" do it!
%
% Methods in the FACTORIZE class are the same as above except that
% "factorize1" is replaced with "factorize", and the following additional
% methods are added:
%
%       c = S(:,1) ;        % returns the 1st column of inv(A)
%       [m n] = size(F) ;   % returns the size of F or S
%
% S=inverse(F) simply flags S as the factorization of the inverse of A,
% without actually computing the inverse.  Then x=S*b solves a linear
% system using forward and back subsitution, rather than multiplying by the
% inverse.
%
% Suppose you want to solve two linear systems, A*x=b and A*y=c.  Do not be
% tempted to compute the inverse.  You can factorize the matrix (LU,
% Cholesky, or QR, depending on the matrix), and then use the factors
% twice, but it can be hard to remember all the different formulas.  You
% can do x=A\b and y=A\c, but this requires A to be factorized twice.
% Instead, with the inverse function defined as S=inverse(factorize(F)),
% you can use Method 1 instead.
%
% Method 1, fast and accurate:
%
%       S = inverse(A) ;                % factorize A, so that P*A=L*U
%       x = S*b ;                       % solve A*x=b for x
%       y = S*c ;                       % solve A*y=c for y
%
% Method 2 is equivalent to Method 1:
%
%       F = factorize1(A) ;             % factorize A, so that P*A=L*U
%       x = F\b ;                       % solve A*x=b for x
%       y = F\c ;                       % solve A*y=c for y
%
% Method 3 is the same, but hard to remember and synatically ugly:
%
%       [L,U,p] = lu(A,'vector') ;      % factorize A(:,p) = L*U
%       x = U \ (L \ (b (p,:)) ;        % solve A*x=b for x
%       y = U \ (L \ (c (p,:)) ;        % solve A*x=b for c
%
% Method 4 is accurate and simple, but it's slower than the methods above:
%
%       x = A\b ;                       % solve A*x=b for x
%       y = A\c ;                       % solve A*y=c for y
%
% Never use Method 5 (both slow and inaccurate):
%
%       S = inv(A) ;                    % compute the inverse of A (ack!)
%       x = S*b ;                       % solve A*x=b for x
%       y = S*c ;                       % solve A*y=c for y
%
% Example
%
%       F = factorize1 (A) ; x = F\b ;  % factorize once, and reuse it
%       x = inverse(A)*b ;              % solve A*x=b using a factorization
%
% See also mldivide, mrdivide, mtimes, inverse, factorize
% Do not see inv!

% Copyright 2009, Timothy A. Davis, University of Florida

properties (SetAccess = protected)
    % the data in the factorize1 object, A(p,:) = L*U.
    A, L, U, p, is_inverse = false ;
end

methods

    function F = factorize1 (A)
        % constructor: computes a factorization of A
        [m n] = size (A) ;
        if (ndims (A) > 2 || m ~= n)
            error ('Matrix must be square and 2D.') ;
        end
        F.A = A ;
        [F.L F.U F.p] = lu (A, 'vector') ;  % gives A(p,:) = L*U
        if (nnz (diag (F.U)) ~= n)
            error ('Matrix is rank deficient.') ;
        end
    end

    function x = mldivide (F,b)
        % mldivide: x = A\b using the factorization of A
        if (F.is_inverse)
            x = F.A * b ;       % x = inverse(A)\b is just A*b
        else
            x = F.U \ (F.L \ b (F.p,:)) ;
        end
    end

    function x = mrdivide (b,F)
        % mrdivide: x = b/A using the factorization of A
        if (F.is_inverse)
            x = b * F.A ;       % x = b/inverse(A) is just b*A
        else
            x = (b / F.U) / F.L ;
            x (:,F.p) = x ;
        end
    end

    function disp (F)
        % disp: displays F
        fprintf ('  A: \n') ; disp (F.A) ;
        fprintf ('  L: \n') ; disp (F.L) ;
        fprintf ('  U: \n') ; disp (F.U) ;
        fprintf ('  p: \n') ; disp (F.p) ;
        fprintf ('  is_inverse: %d\n', F.is_inverse) ;
        fprintf ('  LU factorization: A(p,:) = L*U\n') ;
    end

    function F = inverse (F)
        % inverse: "inverts" F by flagging it as factorization of inv(A)
        F.is_inverse = ~(F.is_inverse) ;
    end

    function x = mtimes (y,z)
        % mtimes: A*b, inv(A)*b, b*A, or b*inv(A), without computing inv(A)
        if (isa (y, 'factorize1'))
            if (y.is_inverse)
                F = y ;                 % x = inv(A)*b via x = A\b
                b = z ;
                x = F.U \ (F.L \ b (F.p,:)) ;
            else
                x = y.A * z ;           % x = A*b
            end
        else
            if (z.is_inverse)
                F = z ;                 % x = b*inv(A) via x = b/A
                b = y ;
                x = (b / F.U) / F.L ;
                x (:,F.p) = x ;
            else
                x = y * z.A ;           % x = b*A
            end
        end
    end

    function S = double (F)
        % double: returns the factorization as a single matrix, A or inv(A)
        if (F.is_inverse)
            % ack!  The explicit inverse has been requested...
            S = mldivide (F, eye (size (F.A,1))) ;
        else
            % F represents the factorization of A (not its inverse), so
            % just return A itself.
            S = F.A ;
        end
    end
end
end
