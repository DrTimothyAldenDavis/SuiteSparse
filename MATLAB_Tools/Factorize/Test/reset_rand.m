function reset_rand
%RESET_RAND resets the state of rand
%
% Example
%   reset_rand
%
% See also RandStream

% Copyright 2011, Timothy A. Davis, University of Florida.

reset (RandStream.getDefaultStream) ;
