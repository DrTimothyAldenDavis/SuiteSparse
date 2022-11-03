function c = cond1est (A)
%COND1EST 1-norm condition estimate.
% Example:
%   c = cond1est(A)
% See also: cs_demo

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

[m n] = size (A) ;
if (m ~= n | ~isreal (A))                                                   %#ok
    error ('A must be square and real') ;
end
if isempty(A)
    c = 0 ;
    return ;
end
[L,U,P,Q] = lu (A) ;
if (~isempty (find (abs (diag (U)) == 0)))                                  %#ok
    c = Inf ;
else
    c = norm (A,1) * norm1est (L,U,P,Q) ;
end
