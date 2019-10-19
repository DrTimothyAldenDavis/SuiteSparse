function Match = maxtrans (A) ;
%
% MAXTRANS: finds a permutation of the columns of a sparse
% matrix so that it has a zero-free diagonal. 
%
% Usage:  Match = maxtrans (A)
%
%	Match = maxtrans (A) ;	% has entries in the range 1:n and -(1:n)
%	p = abs (Match) ;	% the permutation (either full rank or singular)
%	B = A (:, p) ;		% permuted matrix (either full rank or singular)
%	find (Match < 0) ;	% gives a list of zero diagonal entries of B
%	sum (Match > 0) ;	% same as "sprank (A)"
%
% This behaviour differs from p = dmperm (A) in MATLAB, which returns p(i)=0
% if row i is unmatched.  Thus:
%
%	p = dmperm (A) ;	% has entries in the range 0:n
%	p                       % the permutation (only if full rank)
%	B = A (:, p) ;		% permuted matrix (only if full rank)
%	find (p == 0) ;		% gives a list of zero diagonal entries of B
%	sum (p > 0) ;		% definition of sprank (A)
%
% See also: strongcomp, dmperm 
%
% Copyright (c) 2004.  Tim Davis, May, 2004, University of Florida,
% with support from Sandia National Laboratories.  All Rights Reserved.

error ('maxtrans mexfunction not found') ;
