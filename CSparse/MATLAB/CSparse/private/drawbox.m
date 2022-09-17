function drawbox (r1,r2,c1,c2,color,w,e)
%DRAWBOX draw a box around a submatrix in the figure.
%   Used by cspy, cs_dmspy, and ccspy.
%   Example:
%       drawbox (r1,r2,c1,c2,color,w,e)
%   See also drawboxes, plot

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

if (r1 == r2 | c1 == c2)                                                    %#ok
    return
end

if (e == 1)
    r1 = r1 - .5 ;
    r2 = r2 - .5 ;
    c1 = c1 - .5 ;
    c2 = c2 - .5 ;
else
    r1 = ceil (r1 / e) - .5 ;
    r2 = ceil ((r2 - 1) / e) + .5 ;
    c1 = ceil (c1 / e) - .5 ;
    c2 = ceil ((c2 - 1) / e) + .5 ;
end

if (c2 > c1 | r2 > r1)                                                      %#ok
    plot ([c1 c2 c2 c1 c1], [r1 r1 r2 r2 r1], color, 'LineWidth', w) ;
end
