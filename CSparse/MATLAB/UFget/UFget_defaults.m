function params = UFget_defaults
%UFGET_DEFAULTS returns default parameter settings for UFget.
%   Usage:  params = UFget_defaults ; Returns the default parameter settings for
%   UFget.  This file may be editted to change these settings:
%
%   params.url: UF sparse matrix web site
%   params.dir: your local directory for downloaded sparse matrices.
%   params.refresh:  how many days should elapse before re-downloading the
%       index file (for obtaining access to new matrices in the collection).
%
%   Example:
%       params = UFget_defaults ;
%
%   See also UFget.

%   Copyright 2008, Tim Davis, University of Florida.

%-------------------------------------------------------------------------------
% location of the UF sparse matrix collection
params.url = 'http://www.cise.ufl.edu/research/sparse/mat' ;

%-------------------------------------------------------------------------------
% decode the current directory for this M-file
s = which (mfilename) ;
i = find (s == filesep) ;
s = s (1:i(end)) ;

%-------------------------------------------------------------------------------
% define the directory to download into.  Should end in file separator.
% Some examples include:
% params.dir = '/cise/research/sparse/public_html/mat/' ;
% params.dir = 'your directory here/' ;
% params.dir = 'c:\matlab\work\UFget\mat\' ;

% Default: the directory containing this UFget_defaults function
params.dir = sprintf ('%smat%s', s, filesep) ;

%-------------------------------------------------------------------------------
% define how often to check for a new index file (in # of days)
% inf will force the program to ignore the need to refresh

% params.refresh = Inf ;
params.refresh = 90 ;
