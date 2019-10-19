if (0)
    load west0479
    A = west0479 ;
    n = size (A,1) ;
    b = rand (n,1) ;
    [x info] = klus (A,b) ;
    info
    [x info] = klus (A,b,0) ;
    info
end


if (0)
    load /cise/homes/davis/zzz/largematrix
    A = spconvert (largematrix) ;
    spy (A) ;
    n = size (A,1) ;
    b = rand (n,1) ;
    [x info] = klus (A,b) ;
    info
end

load /cise/homes/davis/zzz/newtest
A = spconvert (newtest) ;
spy (A) ;
n = size (A,1) ;
b = ones (n,1) ;
[x info] = klus (A,b) ;
info

norm (A*x-b)

figure (2)
subplot (1,2,1) ; spy (A) ; title ('A') ;
[p,q,r,s] = dmperm (A) ;
C = A(p,q) ;
subplot (1,2,2) ; spy (C) ; title ('C') ;

nblocks = length (r)-1 ;
for k = 1:nblocks
    k1 = r (k) ;
    k2 = r (k+1)-1 ;

    B = C (k1:k2, k1:k2) ;

    nk = k2-k1+1 ;
    if (nk > 1)
 	fprintf ('%3d: %4d cond %g\n', k, k2-k1+1, cond (full (B))) ;
    end

end
