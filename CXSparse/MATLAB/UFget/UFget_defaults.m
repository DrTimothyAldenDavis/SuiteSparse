function params = UFget_defaults
%UFGET_DEFAULTS returns default parameter settings for UFget.
%   Usage:  params = UFget_defaults ; Returns the default parameter settings for
%   UFget.  Edit the UFget/UFsettings.txt file to change these settings.
%
%   params.url: URL for *.mat files
%   params.dir: your local directory for downloaded *.mat files
%   params.refresh:  how many days should elapse before re-downloading the
%       index file (for obtaining access to new matrices in the collection).
%   params.topurl: URL for UF Sparse Matrix Collection
%   params.topdir: your directory for mat/, matrices/, MM/ and RB/ directories
%
%   Example:
%       params = UFget_defaults ;
%
%   See also UFget.

% Copyright 2009-2012, Timothy A. Davis, http://www.suitesparse.com

% decode the current directory for this M-file
s = which (mfilename) ;
i = find (s == filesep) ;
this = s (1:i(end)) ;

% defaults, if UFsettings.txt is not present
params.topdir = '' ;
params.topurl = 'http://www.cise.ufl.edu/research/sparse' ;
params.refresh = 30 ;

% open the UFsettings.txt file, if present, and read the default settings

f = -1 ;
try
    f = fopen (sprintf ('%sUFsettings.txt', this), 'r') ;
    if (f >= 0)
        % get the location of the mat directory
        s = fgetl (f) ;
        if (ischar (s))
            params.topdir = s ;
        end
        % get the default URL
        s = fgetl (f) ;
        if (ischar (s))
            params.topurl = s ;
        end
        % get the refresh rate
        s = fgetl (f) ;
        if (ischar (s) && ~isempty (s))
            params.refresh = str2double (s) ;
        end
    end
catch
end

try
    if (f >= 0)
        fclose (f) ;
    end
catch
end

if (isempty (params.topdir))
    params.topdir = this ;
end

if (params.topdir (end) ~= filesep)
    params.topdir = [params.topdir filesep] ;
end

params.url = [params.topurl '/mat'] ;
params.dir = sprintf ('%smat%s', params.topdir, filesep) ;
