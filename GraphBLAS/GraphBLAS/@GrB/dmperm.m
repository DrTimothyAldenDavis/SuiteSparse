function [p, varargout] = dmperm (G)
%DMPERM Dulmage-Mendelsohn permutation.
% See 'help dmperm' for details.
%
% See also GrB/amd, GrB/colamd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[p, varargout{1:nargout-1}] = builtin ('dmperm', logical (G)) ;

