%UFget_example is a demo for UFget.
%   This example script gets the index file of the UF sparse matrix collection,
%   and then loads in all symmetric non-binary matrices, in increasing order of
%   number of rows in the matrix.
%
%   Example:
%       type UFget_example ; % to see an example of how to use UFget
%
%   See also UFget, UFweb.

%   Copyright 2005, Tim Davis, University of Florida.

%  Nov 3, 2005.

type UFget_example ;

index = UFget ;
f = find (index.numerical_symmetry == 1 & ~index.isBinary) ;
[y, j] = sort (index.nrows (f)) ;
f = f (j) ;

for i = f
    fprintf ('Loading %s%s%s, please wait ...\n', ...
	index.Group {i}, filesep, index.Name {i}) ;
    Problem = UFget (i,index)
    spy (Problem.A) ;
    title (sprintf ('%s:%s', Problem.name, Problem.title')) ;
    UFweb (i) ;
    input ('hit enter to continue:') ;
end

