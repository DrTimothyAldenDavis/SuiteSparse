function s = UFint (x)
%UFINT print an integer to a string, adding commas every 3 digits
% If negative, the result is the single character '-'.
%
% Example:
%   UFint (-2)
%   UFint (2^30)
%
% See also sprintf.

% Copyright 2006-2007, Timothy A. Davis

if (x < 0)
    s = '-' ;
    return
end
t = sprintf ('%d', fix (x)) ;
len = length (t) ;
if (len <= 3 || len > 12 || ~isempty (strfind (t, 'e')))
    s = t ;
elseif (len == 4)
    s = [ t(1) ',' t(2:4) ] ;
elseif (len == 5)
    s = [ t(1:2) ',' t(3:5) ] ;
elseif (len == 6)
    s = [ t(1:3) ',' t(4:6) ] ;
elseif (len == 7)
    s = [ t(1) ',' t(2:4) ',' t(5:7) ] ;
elseif (len == 8)
    s = [ t(1:2) ',' t(3:5) ',' t(6:8) ] ;
elseif (len == 9)
    s = [ t(1:3) ',' t(4:6) ',' t(7:9) ] ;
elseif (len == 10)
    s = [ t(1) ',' t(2:4) ',' t(5:7) ',' t(8:10) ] ;
elseif (len == 11)
    s = [ t(1:2) ',' t(3:5) ',' t(6:8) ',' t(9:11) ] ;
elseif (len == 12)
    s = [ t(1:3) ',' t(4:6) ',' t(7:9) ',' t(10:12) ] ;
end

