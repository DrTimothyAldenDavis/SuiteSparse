function print_resid (A, x, b)
% print_resid (A, x, b), print the relative residual,
% norm (A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))
% Example:
%   print_resid (A, x, b) ;
% See also: cs_demo

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

fprintf ('resid: %8.2e\n', ...
    norm (A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))) ;
