function [L,U,Off,Pnum,R] = kluf (A, tol,P,Q,R,Lnz,Info)

% Factor A using symbolic info from klua.
%
% [P,Q,R,Lnz,Info1] = klua (A) ;
% [L,U,Off,Pnum,Rs,Info2] = kluf (A, P,Q,R,Lnz,Info1, Control) ;
%
% The factorization is L*U + Off = Rs (Pnum,Pnum) \ (A (Pnum,Q)), where Rs is
% a diagonal matrix of row scale factors.  If Pnum and Q are converted to
% permutation matrices, then L*U + Off = Pnum * (Rs\A) * Q. 


error ('kluf mexfunction not found') ;

