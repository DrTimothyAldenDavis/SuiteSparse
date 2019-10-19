function [p,q,r,lnz] = btfamd (A)
% [p,q,r,lnz] = btfamd (A)
%
% does the same thing as klua mexFunction

[m n] = size (A) ;
if (m ~= n | sprank (A) < n | ~issparse (A))
    fprintf ('solver: A must be sparse, square, non-singular\n') ;
    return
end

% put in upper block triangular form
[cp,cq,r] = btf (A) ;

A = A (cp,cq) ;
nblocks = length (r) - 1 ;

lnz = -ones (1, nblocks) ;
p = -ones (1, n) ;
q = -ones (1, n) ;

for k = 1:nblocks

    % get the kth block: scale (A (k1:k2, k1:k2))
    k1 = r (k) ;
    k2 = r (k+1) - 1 ;
    nk = k2 - k1 + 1 ;

    % order the block
    [ap, Info] = amd (A (k1:k2, k1:k2)') ;
    lnz (k) = Info (10) + nk ;

    % combine the permutations
    for kk = 1:nk
	q (k1:k2) = cq (ap + k1 - 1) ;
	p (k1:k2) = cp (ap + k1 - 1) ;
    end

end

