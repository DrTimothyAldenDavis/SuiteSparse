function list = UFgrep (expression, index)
%UFGREP search for matrices in the UF Sparse Matrix Collection.
% UFgrep returns a list of Problem id's whose Problem.name string matches an
% expression.  With no output arguments, the list is displayed.  Otherwise, it
% is returned as a list of integer id's.
%
% Example:
%   UFgrep ('HB') ;         % all matrices in the HB group
%   UFgrep ('\<B*') ;       % all matrices in groups starting with the letter B
%   UFgrep ('bp') ;         % all bp matrices (in the HB group)
%   UFgrep ('[0-9]') ;      % all matrices with a digit in their name
%   UFgrep ('c-') ;         % all c-* optimization matrices (in 2 groups)
%
% An optional 2nd input argument is the UF index, which is a little faster if
% UFgrep is used multiple times.  With no input arguments, all Problems are
% listed.
%
%   index = UFget ;
%   UFgrep ('HB/*', index) ;
%
% See also regexp, UFget.

% Copyright 2008, Timothy A. Davis

if (nargin < 2)
    index = UFget ;
end

if (nargin < 1)
    expression = '.' ;
end

nmat = length (index.nrows) ;
list1 = zeros (1, nmat) ;
matched = 0 ;

for id = 1:nmat
    name = [index.Group{id} '/' index.Name{id}] ;
    if (~isempty (regexp (name, expression, 'once')))
	matched = matched + 1 ;
	list1 (matched) = id ;
    end
end

list1 = list1 (1,1:matched) ;

if (nargout == 0)
    for id = list1
	fprintf ('%4d: %s/%s\n', id, index.Group {id}, index.Name {id}) ;
    end
else
    list = list1 ;
end
