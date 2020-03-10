function centrality = gap_centrality (sources, A, AT)
%GAP_CENTRALITY batch betweenness centrality of a graph, via GraphBLAS
%
% Given a set of source nodes s (an array of integers in the range 1 to n) and
% an adjacency matrix A, c=gap_centrality(s,A) computes the betweenness
% centrality of all nodes in the graph.  The result is a vector c of size n.
% The centrality of a node is the relative number of shortest paths that pass
% through node i.
%
% Let sigma(s,t|i) be the total number of shortest paths from node s, for s in
% the list soources, to node t, that pass through node i.  Let sigma(s,t) be
% the number total number of shortest paths from s to t.  Then c(i) is the sum
% of sigma(s,t|i)/sigma(s,t), for all unique s and t (s is not t) that are also
% not equal to i.
%
% A must be square.  It may be unsymmetric, and self-edges (diagonal entries)
% are OK.  GrB.format must be stored 'by row'.  AT is optional, but if present,
% it must be the transpose of A (and must also be stored by row).  If not
% present, AT=A' is computed first.  The values of A and AT are ignored; just
% the pattern of the two matrices is important.
%
% The list of sources should be small; length(sources) == 4 is typical.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

% get input problem size
[m,n] = size (A) ;
ns = length (sources) ;

if (m ~= n)
    error ('A must be square') ;
end
if (~isequal (GrB.format (A), 'by row'))
    % FUTURE: handle the case when A is stored by column
    error ('A must be a GrB matrix stored by row') ;
end
if (nargin < 3)
    % transpose not provided, so compute it
    AT = GrB.trans (A, struct ('format', 'by row')) ;
elseif (~isequal (GrB.format (AT), 'by row'))
    error ('AT must be a GrB matrix stored by row') ;
end

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

% create result, and workspace
paths      = GrB (ns, n, 'single', 'by row') ;
frontier   = GrB (ns, n, 'single', 'by row') ;

% paths is a dense matrix
paths (:,:) = 0 ;

% create the initial frontier
for i = 1:ns
    paths (i, sources (i)) = 1 ;
    frontier (i, sources (i)) = 1 ;
end

% descriptors
desc_rc = struct ('out', 'replace', 'mask', 'complement') ;
desc_rs = struct ('out', 'replace', 'mask', 'structural') ;
desc_t0 = struct ('in0', 'transpose') ;

% initial frontier:  frontier<!paths> = frontier*A
frontier = GrB.mxm (frontier, paths, '+.first.single', frontier, A, desc_rc) ;

% S = cell array of frontiers, at each level
S = cell (1, n) ;

%-------------------------------------------------------------------------------
% breadth-first search stage
%-------------------------------------------------------------------------------

for depth = 1:n
    % S {depth} = pattern of frontier
    S {depth} = spones (frontier, 'logical') ;
    % accumulate path counts: paths += frontier
    paths = GrB.assign (paths, '+.single', frontier) ;
    % update frontier: frontier<!paths> = frontier*A
    frontier = GrB.mxm (frontier, paths, '+.first.single', frontier, A, ...
        desc_rc) ;
    % break if frontier is empty
    if (GrB.entries (frontier) == 0)
        break ;
    end
end

clear frontier

%-------------------------------------------------------------------------------
% betweenness centrality computation phase
%-------------------------------------------------------------------------------

% bc_update = ones (ns,n) ;
bc_update = GrB (ns, n, 'single', 'by row') ;
bc_update (:,:) = 1 ;

% W = empty ns-by-n workspace
W = GrB (ns, n, 'single', 'by row') ;

% backtrack through the BFS levels, and compute centrality update for each node
for i = depth:-1:2
    % add contributions by successors and mask with that level's frontier
    % W<S{i}> = bc_update ./ path
    W = GrB.emult (W, S{i}, '/', bc_update, paths, desc_rs) ;
    % W<S{i-1}> = W*A'
    W = GrB.mxm (W, S{i-1}, '+.first.single', W, AT, desc_rs) ;
    % bc_update += W .* paths
    bc_update = GrB.emult (bc_update, '+', W, '*', paths) ;
end

% initialize centrality with -ns to avoid counting zero-length paths
centrality = GrB (n, 1, 'single', 'by col') ;
centrality (:) = -ns ;

% centrality (i) += sum (bc_update (:,i)) for all nodes i
% centrality = centrality + sum (bc_update, 1) ;
centrality = GrB.vreduce (centrality, '+', '+', bc_update, desc_t0) ;

