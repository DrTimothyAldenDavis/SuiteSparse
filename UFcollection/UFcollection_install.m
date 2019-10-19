%UFcollection_install: install the UFcollection toolbox
%
% Example:
%   UFcollection_install
%
% See also UFget.

% Copyright 2006, Timothy A. Davis

if (~isempty (strfind (computer, '64')))
    error ('64-bit version not yet supported') ;
end

mex -I../UFconfig UFfull_write.c
addpath (pwd) ;

fprintf ('The UFcollection toolbox is now installed.  Your path has been\n') ;
fprintf ('temporarily modified by adding the directory:\n') ;
fprintf ('%s\n', pwd) ;
fprintf ('Use pathtool to add it permanently\n') ;
