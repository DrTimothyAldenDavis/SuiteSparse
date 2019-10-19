% UFcollection: software for managing the UF Sparse Matrix Collection
%
% To create the index:
%
%   UFindex    - create the index for the UF Sparse Matrix Collection
%   UFstats    - compute matrix statistics for the UF Sparse Matrix Collection
%
% To create the web pages:
%
%   UFallpages - create all web pages for the UF Sparse Matrix Collection
%   UFgplot    - draw a plot of the graph of a sparse matrix
%   UFint      - print an integer to a string, adding commas every 3 digits
%   UFlist     - create a web page index for the UF Sparse Matrix Collection
%   UFlists    - create the web pages for each matrix list (group, name, etc.)
%   UFlocation - URL and top-level directory of the UF Sparse Matrix Collection
%   UFpage     - create web page for a matrix in UF Sparse Matrix Collection
%   UFpages    - create web page for each matrix in UF Sparse Matrix Collection
%   dsxy2figxy - Transform point or position from axis to figure coords
%
% To create the Matrix Market and Rutherford/Boeing versions of the collection:
%
%   UFexport     - export to Matrix Market and Rutherford/Boeing formats
%   UFread       - read a Problem in Matrix Market or Rutherford/Boeing format
%   UFwrite      - write a Problem in Matrix Market or Rutherford/Boeing format
%   UFfull_read  - read a full matrix using a subset of Matrix Market format
%   UFfull_write - write a full matrix using a subset of Matrix Market format
%
% Other:
%
%   UFcollection_install - install the UFcollection toolbox
%
% Example:
%   UFindex       % create index (UF_Index.mat) for use by UFget
%   UFallpages    % create all web pages for the UF Sparse Matrix Collection
%
% Requires UFget, CSparse, CHOLMOD, AMD, COLAMD, METIS.

% Copyright 2006-2007, Timothy A. Davis

