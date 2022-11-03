function g = givens2(a,b)
%GIVENS2 find a Givens rotation.
% Example:
%   g = givens2(a,b)
% See also: cs_demo

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

if (b == 0)
    c = 1 ; s = 0 ;
elseif (abs (b) > abs (a))
    tau = -a/b ; s = 1 / sqrt (1+tau^2) ; c = s*tau ;
else
    tau = -b/a ; c = 1 / sqrt (1+tau^2) ; s = c*tau ;
end
g = [c -s ; s c] ;
