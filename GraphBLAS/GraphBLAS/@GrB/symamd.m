function [p, varargout] = symamd (G, varargin)
%SYMAMD approximate minimum degree ordering.
% See 'help symamd' for details.
%
% See also GrB/amd, GrB/colamd, GrB/symrcm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[p, varargout{1:nargout-1}] = symamd (double (G), varargin {:}) ;

