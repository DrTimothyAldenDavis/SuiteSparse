function checkbtf (A, p, q, r)
%CHECKBTF ensure A(p,q) is in BTF form
%
% A(p,q) is in BTF form, r the block boundaries
%
% Example:
%   [p,q,r] = dmperm (A)
%   checkbtf (A, p, q, r)
%
% See also drawbtf, maxtrans, strongcomp.

% BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
% Author: Timothy A. Davis.
% SPDX-License-Identifier: LGPL-2.1+

[m n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end

if (any (sort (p) ~= 1:n))
    error ('p not a permutation') ;
end

if (any (sort (q) ~= 1:n))
    error ('q not a permutation') ;
end

nblocks = length (r) - 1 ;

if (r (1) ~= 1)
    error ('r(1) not one') ;
end

if (r (end) ~= n+1)
    error ('r(end) not n+1') ;
end

if (nblocks < 1 | nblocks > n)                                              %#ok
    error ('nblocks wrong') ;
end

nblocks = length (r) - 1 ;
rdiff = r (2:(nblocks+1)) - r (1:nblocks) ;
if (any (rdiff < 1) | any (rdiff > n))                                      %#ok
    error ('r bad')
end

