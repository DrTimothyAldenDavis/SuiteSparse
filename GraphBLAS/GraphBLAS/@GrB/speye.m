function C = speye (varargin)
%GRB.SPEYE sparse identity matrix.
% C = GrB.speye (...) is identical to GrB.eye; see 'help GrB.eye' for
% details.
%
% See also GrB.eye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB (gb_speye ('speye', varargin {:})) ;

