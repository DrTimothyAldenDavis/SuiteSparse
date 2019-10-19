function x = mtimes (y,z)
%MTIMES A*b, inv(A)*b, b*A, or b*inv(A), without computing inv(A)
%
% Example
%   S = inverse (A) ;       % a factorized representation of inv(A)
%   x = S*b ;               % same as x=A\b.  Does not use inv(A)
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

if (isa (y, 'factorize'))
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

