function [p, varargout] = colamd (G, varargin)
%COLAMD column approximate minimum degree ordering.
% See 'help colamd' for details.
%
% See also GrB/amd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[p, varargout{1:nargout-1}] = colamd (double (G), varargin {:}) ;

