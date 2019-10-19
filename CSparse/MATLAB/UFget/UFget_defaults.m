function params = UFget_defaults
%UFget_defaults returns default parameter settings for UFget.
%   Usage:  params = UFget_defaults ; Returns the default parameter settings for
%   UFget.  This file may be editted to change these settings:
%
%   params.url: UF sparse matrix web site
%   params.dir: your local directory for downloaded sparse matrices.
%   params.refresh:  how many days should elapse before re-downloading the
%	index file (for obtaining access to new matrices in the collection).
%
%   See also UFget.

%   Copyright 2005, Tim Davis, University of Florida.

%   Modification History:
%   10/11/2001: Created by Erich Mirabal
%   3/12/2002: V1.0 released
%   11/2005: updated for MATLAB 7.1

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
% params.dir = '/cise/research/sparse/public_html/mat/' ;   % if at UF
% params.dir = 'your directory here/' ;
% params.dir = 'c:\matlab\work\UFget\mat\' ;

% Default: the directory containing this UFget_defaults function
params.dir = sprintf ('%smat%s', s, filesep) ;

%-------------------------------------------------------------------------------
% define how often to check for a new index file (in # of days)
% inf will force the program to ignore the need to refresh

% params.refresh = Inf ;				    % if at UF
params.refresh = 90 ;
