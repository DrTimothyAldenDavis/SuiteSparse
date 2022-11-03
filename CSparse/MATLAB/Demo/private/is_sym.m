function sym = is_sym (A)
% sym = is_sym(A)
% 1 if A is square and upper tri., -1 if square and lower tri., 0 otherwise
%
% Example:
%   sym = is_sym (A) ;
% See also: cs_demo

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

[m n] = size (A) ;
sym = 0 ;
if (m == n)
    is_upper = nnz (tril (A,-1)) == 0 ;
    is_lower = nnz (triu (A,1)) == 0 ;
    if (is_upper)
        sym = 1 ;
    elseif (is_lower)
        sym = -1 ;
    end
end

