% @FACTORIZE: an object-oriented method for solving linear system, solving
% least-square problems, and for efficient computation of mathematical
% expressions that use inv(A).
%
%   disp      - displays the factorization F
%   double    - returns the factorization as a single matrix, A or inv(A)
%   end       - returns index of last item for use in subsref
%   factorize - an object-oriented method for solving linear systems
%   inverse   - "inverts" F by flagging it as the factorization of inv(A).
%   mldivide  - x = A\b using the factorization F = factorize(A)
%   mrdivide  - x = b/A using the factorization F = factorize(A)
%   mtimes    - A*b, inv(A)*b, b*A, or b*inv(A), without computing inv(A)
%   size      - returns the size of the matrix F.A in the factorization F
%   subsref   - A(i,j) or (i,j)th entry of inv(A) if F is inverted.
%   plus      - update a dense Cholesky factorization
%   minus     - downdate a dense Cholesky factorization
%
% Example
%
%   F = factorize(A) ;
%   x = F\b ;           % same as x=A\b

% Copyright 2009, Timothy A. Davis, University of Florida

