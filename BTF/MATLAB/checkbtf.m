function checkbtf (A, p, q, r)
% checkbtf (A, p, q, r)
%
% A(p,q) is in BTF form, r the block boundaries
%
% [p,q,r] = dmperm (A) for example

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
    r
    error ('r(1) not one') ;
end

if (r (end) ~= n+1)
    r
    error ('r(end) not n+1') ;
end

if (nblocks < 1 | nblocks > n)
    nblocks
    n
    error ('nblocks wrong') ;
end

nblocks = length (r) - 1 ;
rdiff = r (2:(nblocks+1)) - r (1:nblocks) ;
if (any (rdiff < 1) | any (rdiff > n))
    error ('r bad')
end

