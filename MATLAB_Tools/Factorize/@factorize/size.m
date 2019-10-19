function [m n] = size (F,k)
%SIZE returns the size of the matrix F.A in the factorization F
%
% Example
%   F = factorize (A)
%   size(F)                 % same as size (A)
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

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

