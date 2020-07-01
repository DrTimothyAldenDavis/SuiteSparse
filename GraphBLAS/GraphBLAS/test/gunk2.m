clear all
ver
A = 1287128410976072704
whos
fprintf ('A:         %30o\n', A) ;
C = bitcmp (A, 'uint64')
fprintf ('bitcmp(A): %30o\n', C) ;

