function drawbtf (A, p, q, r)
% drawbtf (A, p, q, r)
%
% A(p,q) is in BTF form, r the block boundaries
%
% [p,q,r] = dmperm (A) for example

nblocks = length (r) - 1 ;

hold off
spy (A (p,q))
hold on

for k = 1:nblocks
    k1 = r (k) ;
    k2 = r (k+1) ;
    nk = k2 - k1 ;
    if (nk > 1)
	plot ([k1 k2 k2 k1 k1]-.5, [k1 k1 k2 k2 k1]-.5, 'g') ;
    end
end
