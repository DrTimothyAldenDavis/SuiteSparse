function dmspy_test
%DMSPY_TEST test cspy, cs_dmspy, and cs_dmperm
% Example:
%   dmspy_test
% See also: testall

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse


index = UFget ;
f = find (index.nblocks > 1) ;
% f = find (index.nblocks > 1 & index.nrows == index.ncols & ...
%    index.nnzdiag == index.nrows) ;
[ignore i] = sort (index.nnz (f)) ;
f = f (i) ;

for i = f
    Prob = UFget (i,index) ;
    disp (Prob) ;
    clf
    subplot (2,2,1) ; cspy (Prob.A) ;
    subplot (2,2,2) ; cs_dmspy (Prob.A) ;
    [p,q,r,s,cc,rr] = cs_dmperm (Prob.A) ;				    %#ok
    subplot (2,2,3) ; plot (p) ;
    subplot (2,2,4) ; plot (q) ;
    drawnow
end
