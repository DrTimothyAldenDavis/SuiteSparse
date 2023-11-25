function [parent, post] = etree2 (A, mode)                            %#ok
%ETREE2  sparse elimination tree.
% Finds the elimination tree of A, A'*A, or A*A', and optionaly
% postorders the tree.  parent(j) is the parent of node j in the tree, or
% 0 if j is a root.  The symmetric case uses only the upper or lower
% triangular part of A (etree2(A) uses the upper part, and etree2(A,'lo')
% uses the lower part).
%
% Example:
%   parent = etree2 (A)         finds the etree of A, using triu(A)
%   parent = etree2 (A,'sym')   same as etree2(A)
%   parent = etree2 (A,'col')   finds the etree of A'*A
%   parent = etree2 (A,'row')   finds the etree of A*A'
%   parent = etree2 (A,'lo')    finds the etree of A, using tril(A)
%
% [parent,post] = etree2 (...) also returns a post-ordering of the tree.
%
% If you have a fill-reducing permutation p, you can combine it with an
% elimination tree post-ordering using the following code.  Post-ordering
% has no effect on fill-in (except for lu), but it does improve the
% performance of the subsequent factorization.
%
% For the symmetric case, suitable for chol(A(p,p)):
%
%       [parent post] = etree2 (A (p,p)) ;
%       p = p (post) ;
%
% For the column case, suitable for qr(A(:,p)) or lu(A(:,p)):
%
%       [parent post] = etree2 (A (:,p), 'col') ;
%       p = p (post) ;
%
% For the row case, suitable for qr(A(p,:)') or chol(A(p,:)*A(p,:)'):
%
%       [parent post] = etree2 (A (p,:), 'row') ;
%       p = p (post) ;
%
% See also treelayout, treeplot, etreeplot, etree.

 % Copyright 2006-2023, Timothy A. Davis, All Rights Reserved.
 % SPDX-License-Identifier: GPL-2.0+

error ('etree2 mexFunction not found') ;
