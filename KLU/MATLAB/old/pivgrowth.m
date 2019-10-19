
function growth = pivgrowth(A)

    [p,q,r,lnz,Info1] = klua (A) ;
    [l,u,off,pnum,rs,Info2] = kluf (A, p,q,r,lnz,Info1) ;

    A = rs(pnum, pnum) \ A(pnum,q) ;
    growth = ones(Info1(2), 1); 
    [m,n] = size(A) ;
    nblocks = Info1 (4) ;
    for i = 1:nblocks
	start = r (i) ;
        en = r (i+1) - 1 ; 
	for j = 0:en-start
	    col = start + j ;
	    growth(col);
	    growth(col) = max(abs(A(start:en,col))) / max(abs(u(:,col))) ;
	end
    end
    growth = min(growth) ;


