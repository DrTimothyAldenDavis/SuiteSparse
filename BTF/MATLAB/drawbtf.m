function drawbtf (A, p, q, r)
%DRAWBTF plot the BTF form of a matrix
%
% A(p,q) is in BTF form, r the block boundaries
%
% Example:
%   [p,q,r] = dmperm (A)
%   drawbtf (A, p, q, r)
%
% See also btf, maxtrans, strongcomp, dmperm.

% BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
% Author: Timothy A. Davis.
% SPDX-License-Identifier: LGPL-2.1+

nblocks = length (r) - 1 ;

hold off
spy (A (p,abs(q)))
hold on

for k = 1:nblocks
    k1 = r (k) ;
    k2 = r (k+1) ;
    nk = k2 - k1 ;
    if (nk > 1)
        plot ([k1 k2 k2 k1 k1]-.5, [k1 k1 k2 k2 k1]-.5, 'r') ;
    end
end
