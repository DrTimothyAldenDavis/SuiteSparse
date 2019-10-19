
gbmake
clear
A = sparse (rand (4))
hi = 1 ;
lo = -2 ;
C = GB_mex_band (A,lo,hi)
C2 = triu (tril (A,hi), lo) ;
assert (isequal (C,C2))
full (C)
