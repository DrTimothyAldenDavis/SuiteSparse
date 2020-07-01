function [parent, varargout] = etree (G, varargin)
%ETREE elimination tree of a GraphBLAS matrix.
% See 'help etree' for details.
%
% See also GrB/amd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = logical (G) ;
[parent, varargout{1:nargout-1}] = builtin ('etree', G, varargin {:}) ;

