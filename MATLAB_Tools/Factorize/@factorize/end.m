function e = end (F,k,n)
%END returns index of last item for use in subsref
%
% Example
%   S = inverse (A) ;   % compute the factorized representation of inv(A)
%   S (:,end)           % compute just the last column of inv(A) or pinv(A)
%                       % if A is rectangular.
%
% See also factorize.

% Copyright 2009, Timothy A. Davis, University of Florida

if (n == 1)
    e = numel (F.A) ;       % # of elements, for linear indexing
else
    e = size (F,k) ;        % # of rows or columns in A or pinv(A)
end

