function print_order (order)
% print_order(order) prints the ordering determined by the order parameter
% Example:
%   print_order (0)
% See also: cs_demo

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

switch (fix (order))
    case 0
        fprintf ('natural    ') ;
    case 1
        fprintf ('amd(A+A'')  ') ;
    case 2
        fprintf ('amd(S''*S)  ') ;
    case 3
        fprintf ('amd(A''*A)  ') ;
    otherwise
        fprintf ('undefined  ') ;
end
