% BTF ordering toolbox, including test codes
%
% Primary functions:
%
%   maxtrans   - finds a permutation of the columns of a sparse matrix
%   strongcomp - Find a symmetric permutation to upper block triangular form.
%
% helper and test functions:
%
%   checkbtf   - ensure A(p,q) is in BTF form
%   drawbtf    - plot the BTF form of a matrix
%   td         - test script for BTF
%   toobig     - list of matrices that are too big for dmperm
%   trav       - exhaustive test script for BTF
%   dp         - dmperm on a sparse matrix
%
% Example:
%   Match = maxtrans (A)
%   [p,q,r] = strongcomp (A)

% Copyright 2006, Timothy A. Davis, University of Florida
