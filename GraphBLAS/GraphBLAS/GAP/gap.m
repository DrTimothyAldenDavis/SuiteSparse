function gap
%GAP run 5 GAP benchmarks (BFS, PR, BC, TC, SSSP; not CC)
%
% CC has not yet been implemented.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

type gap
% gap_bc         % run centrality for the GAP benchmark
gap_sssp       % run SSSP for the GAP benchmark
gap_pr         % run pagerank for the GAP benchmark
gap_tc         % run tricount for the GAP benchmark
gap_bfs        % run bfs for the GAP benchmark

