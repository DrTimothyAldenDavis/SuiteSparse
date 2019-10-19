function [p, stats] = ccolamd (S, knobs, cmember)			    %#ok
%CCOLAMD constrained column approximate minimum degree permutation.
%    p = CCOLAMD(S) returns the column approximate minimum degree permutation
%    vector for the sparse matrix S.  For a non-symmetric matrix S, S(:,p)
%    tends to have sparser LU factors than S.  chol(S(:,p)'*S(:,p)) also tends
%    to be sparser than chol(S'*S).  p=ccolamd(S,1) optimizes the ordering for
%    lu(S(:,p)).  The ordering is followed by a column elimination tree post-
%    ordering.
%
%    Example:
%            p = ccolamd(S)
%            [p stats] = ccolamd(S,knobs,cmember)
%
%    knobs is an optional one- to five-element input vector, with a default
%    value of [0 10 10 1 0] if not present or empty ([ ]).  Entries not present
%    are set to their defaults.
%
%    knobs(1): if nonzero, the ordering is optimized for lu(S(:,p)).  It will
%       be a poor ordering for chol(S(:,p)'*S(:,p)).  This is the most
%       important knob for ccolamd.
%    knobs(2): if S is m-by-n, rows with more than max(16,knobs(2)*sqrt(n))
%       entries are ignored.
%    knobs(3): columns with more than max(16,knobs(3)*sqrt(min(m,n))) entries
%       are ignored and ordered last in the output permutation (subject to the
%       cmember constraints).
%    knobs(4): if nonzero, aggressive absorption is performed.
%    knobs(5): if nonzero, statistics and knobs are printed.
%
%    Type the command "type ccolamd" for a description of the optional stats
%    output.
%
%    cmember is an optional vector of length n.  It defines the constraints on
%    the column ordering.  If cmember(j)=s, then column j is in constraint set
%    s (s must be in the range 1 to n).  In the output permutation p, all
%    columns in set 1 appear first, followed by all columns in set 2, and so
%    on.  cmember=ones(1,n) if not present or empty.  ccolamd(S,[],1:n) returns
%    1:n.
%
%    p = ccolamd(S) is about the same as p = colamd(S).  knobs and its default
%    values differ.  colamd always does aggressive absorption, and it finds an
%    ordering suitable for both lu(S(:,p)) and chol(S(:,p)'*S(:,p)); it cannot
%    optimize its ordering for lu(S(:,p)) to the extent that ccolamd(S,1) can.
%
%    Authors: S. Larimore, T. Davis (Univ of Florida), and S. Rajamanickam, in
%    collaboration with J. Gilbert and E. Ng.  Supported by the National
%    Science Foundation (DMS-9504974, DMS-9803599, CCR-0203270), and a grant
%    from Sandia National Lab.  See http://www.cise.ufl.edu/research/sparse
%    for ccolamd, csymamd, amd, colamd, symamd, and other related orderings.
%
%    See also AMD, CSYMAMD, COLAMD, SYMAMD, SYMRCM.

% ----------------------------------------------------------------------------
% CCOLAMD version 2.5.
% Copyright 2006, Univ. of Florida.  Authors: Timothy A. Davis,
% Sivasankaran Rajamanickam, and Stefan Larimore
% See lesser.txt for the Version 2.1 of the GNU Lesser General Public License
% http://www.cise.ufl.edu/research/sparse
% ----------------------------------------------------------------------------

%    stats(1): number of dense or empty rows removed prior to ordering
%    stats(2): number of dense or empty columns removed prior to ordering.
%       These columns are placed last in their constraint set.
%    stats(3): number of garbage collections performed.
%    stats (4:7) provide information if CCOLAMD was able to continue.  The
%    matrix is OK if stats (4) is zero, or 1 if invalid.  stats (5) is the
%    rightmost column index that is unsorted or contains duplicate entries,
%    or zero if no such column exists.  stats (6) is the last seen duplicate
%    or out-of-order row index in the column index given by stats (5), or zero
%    if no such row index exists.  stats (7) is the number of duplicate or
%    out-of-order row indices.
%
%    stats (8:20) is always zero in the current version of CCOLAMD (reserved
%    for future use).

error ('ccolamd: mexFunction not found') ;
