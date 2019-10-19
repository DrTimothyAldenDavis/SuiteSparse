function UFexport (list, check, tmp)
%UFEXPORT export to Matrix Market and Rutherford/Boeing formats
%
% Example:
%   UFexport ;          % export the entire collection
%   UFexport (list) ;   % just export matrices whose id's are given in the list
%   UFexport (list, 'check') ;      % also read them back in, to check
%
% If the list is empty, all matrices in the collection are exported.
% A 3rd argument tmp changes the tmp directory for UFread.
%
% See also UFget, UFwrite, RBio, mwrite.


% Copyright 2006-2007, Timothy A. Davis

%-------------------------------------------------------------------------------
% get the input arguments
%-------------------------------------------------------------------------------

index = UFget ;
nmat = length (index.nrows) ;

if (nargin < 1 || isempty (list))
    list = 1:nmat ;
end

check = ((nargin > 1) && strcmp (check, 'check')) ;

if (nargin < 3)
    tmp = '' ;
end

%-------------------------------------------------------------------------------
% determine the top-level directory to use
%-------------------------------------------------------------------------------

[url topdir] = UFlocation ;

%-------------------------------------------------------------------------------
% export the matrices
%-------------------------------------------------------------------------------

for id = list

    % get the MATLAB version
    Problem = UFget (id, index) ;
    disp (Problem) ;

    % create the MM and RB versions
    UFwrite (Problem, [topdir 'MM'], 'MM', 'tar') ;
    UFwrite (Problem, [topdir 'RB'], 'RB', 'tar') ;

    % check the new MM and RB versions
    if (check)
	for format = { 'MM' , 'RB' }
	    try
		if (isempty (tmp))
		    P2 = UFread ([topdir format{1} filesep Problem.name]) ;
		else
		    P2 = UFread ([topdir format{1} filesep Problem.name], tmp) ;
		end
	    catch
		% The Problem may be too large for two copies to be in the
		% MATLAB workspace at the same time.  This is not an error,
		% but it means that the Problem cannot be checked.
		P2 = [ ] ;
		fprintf ('Unable to read %s/%s\n', format {1}, Problem.name) ;
		fprintf ('%s\n', lasterr) ;
	    end
	    if (~isempty (P2) && ~isequal (Problem, P2))
                Problem
                P2
		error ('%s version mismatch: %s\n', format {1}, Problem.name) ;
	    end
	    clear P2
	end
    end
end

