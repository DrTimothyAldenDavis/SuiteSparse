function UFlists
%UFLISTS create the web pages for each matrix list (group, name, etc.)
% Places the web pages in the matrices/ subdirectory of the current directory.
%
% Example:
%   UFlists
%
% See also UFget, UFlist

% Copyright 2006-2007, Timothy A. Davis

% create all the web pages for the lists
UFlist ('group') ;
UFlist ('name') ;
UFlist ('id') ;
UFlist ('dimension') ;
UFlist ('nnz') ;
UFlist ('symmetry') ;
UFlist ('type') ;

% do all the group pages, and the list of groups
index = UFget ;
nmat = length (index.nrows) ;

[ignore, i] = sort (index.Group) ;
g = index.Group (i) ;

% create the primary directory
[url topdir] = UFlocation ;
matrices = [topdir 'matrices'] ;
if (~exist (matrices, 'dir'))
    mkdir (matrices) ;
end

% create the groups.html file
f = fopen ([matrices filesep 'groups.html'], 'w') ;
if (f < 0)
    error ('unable to create groups.html file\n') ;
end

% add the header
fprintf (f, ...
	'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n') ;
fprintf (f, '<html lang="EN"><head>\n') ;
fprintf (f, '<meta http-equiv="content-type" content="text/html; charset=') ;
fprintf (f, 'iso-8859-1"><title>UF Sparse Matrix Collection: group list') ;
fprintf (f, '</title></head>\n') ;
fprintf (f, '<body bgcolor="#ffffff" link="#0021a5">\n') ;
fprintf (f, '<li><a href="./index.html">UF Sparse Matrix Collection</a>\n') ;

fprintf (f, '<p>List of matrix groups in the UF Sparse Matrix Collection:\n') ;
fprintf (f, '<p><table border=1>\n') ;
fprintf (f, '<th>Group</th>\n') ;
fprintf (f, '<th># matrices</th>\n') ;
fprintf (f, '<th>details</th>\n') ;
fprintf (f, '<th>description</th>\n') ;

% find all the groups
group = '' ;
ngroup = 0 ;
for i = 1:nmat
    if (strcmp (group, g {i}))
	continue
    end
    group = g {i} ;
    ngroup = ngroup + 1 ;
    groups {ngroup} = group ;						    %#ok
end

nmat = 0 ;
for i = 1:ngroup
    group = groups {i} ;
    UFlist ('group', group) ;
    fprintf (f, '<tr>\n') ;

    % link to group
    fprintf (f, '<td><a href="%s/index.html">%s</a></td>\n', group, group) ;

    % number of matrices
    d = dir ([topdir 'mat' filesep group filesep '*.mat']) ;
    nmat_group = size (d,1) ;
    fprintf (f, '<td>%d</td>\n', nmat_group) ;
    nmat = nmat + nmat_group ;

    % link to README.txt file ("details")
    f2 = fopen ([topdir filesep 'mat' filesep group filesep 'README.txt'], 'r');
    if (f2 < 0)
	error (['no README file for group: ' group]) ;
    else
	s = fgets (f2) ;
	fprintf (f, ...
	    '<td><a href="../mat/%s/README.txt">details</a></td>\n', group) ;
    end

    % one-line description (first line of README.txt)
    fprintf (f, '<td>%s</td>\n', s) ;
    fclose (f2);

    fprintf (f, '</tr>\n') ;
end
fprintf (f, '</table>\n') ;

fprintf (f, '<p>Total number of matrices in UF Sparse Matrix Collection:') ;
fprintf (f, ' %d\n', nmat) ;
fprintf (f, '<p><p><i>Maintained by Tim Davis, last updated %s.', date) ;
fprintf (f, '<br>Matrix pictures by <a href=') ;
fprintf (f, '"%sCSparse/CSparse/MATLAB/CSparse/cspy.m">cspy</a>, a MATLAB', ...
    url) ;
fprintf (f, ' function in the <a href="%sCSparse">CSparse</a> package.\n', ...
    url) ;
fprintf (f, '</body>\n') ;
fprintf (f, '</html>\n') ;
fclose (f) ;
