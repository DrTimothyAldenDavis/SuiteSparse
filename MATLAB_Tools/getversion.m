function v = getversion
%GETVERSION return MATLAB version number as a double.
% GETVERSION determines the MATLAB version, and returns it as a double.  This
% allows simple inequality comparisons to select code variants based on ranges
% of MATLAB versions.
%
% As of MATLAB 7.5, the version numbers are listed below:
%
%   MATLAB version                      getversion return value
%   -------------------------------     -----------------------
%   7.5.0.342 (R2007b)                  7.5
%   7.4.0.287 (R2007a)                  7.4
%   7.3.0.267 (R2006b)                  7.3
%   7.2.0.232 (R2006a)                  7.2
%   7.1.0.246 (R14) Service Pack 3      7.1
%   7.0.4.365 (R14) Service Pack 2      7.04
%   7.0.1.24704 (R14) Service Pack 1    7.01
%   6.5.2.202935 (R13) Service Pack 2   6.52
%   6.1.0.4865 (R12.1)                  6.1
%   ...
%   5.3.1.something (R11.1)             5.31
%   3.2 whatever                        3.2
%
% Example:
%
%       v = getversion ;
%       if (v >= 7.0)
%           this code is for MATLAB 7.x and later
%       elseif (v == 6.52)
%           this code is for MATLAB 6.5.2
%       else
%           this code is for MATLAB versions prior to 6.5.2
%       end
%
% This getversion function has been tested on versions 6.1 through 7.5, but it
% should work in any MATLAB that has the functions version, sscanf, and length.
%
% See also version, ver, verLessThan.

% Copyright 2007, Timothy A. Davis, Univ. of Florida

% This function does not use ver, in the interest of speed and portability.
% "version" is a built-in that is about 100 times faster than the ver m-file.
% ver returns a struct, and structs do not exist in old versions of MATLAB.
% All 3 functions used here (version, sscanf, and length) are built-in.

v = sscanf (version, '%d.%d.%d') ;
v = 10.^(0:-1:-(length(v)-1)) * v ;
