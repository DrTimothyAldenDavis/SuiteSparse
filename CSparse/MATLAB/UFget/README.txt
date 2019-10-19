UFget:  a MATLAB interface to the UF sparse matrix collection.
MATLAB 7.0 or later is required.

Date: April 1, 2008.

Copyright 2005-2008, Tim Davis, University of Florida.
Authors: Tim Davis and Erich Mirabel.
Availability: http://www.cise.ufl.edu/research/sparse/mat/UFget

See http://www.cise.ufl.edu/research/sparse/mat/UFget.tar.gz
for a single archive file with all the files listed below.

    UFget/Contents.m            help for UFget
    UFget/README.txt            this file

    UFget/UFget_defaults.m      default parameter settings for UFget
    UFget/UFget_example.m       demo for UFget
    UFget/UFget_lookup.m        get the group, name, and id of a matrix
    UFget/UFget.m               primary user interface

    UFget/UFweb.m               opens the URL for a matrix or collection

    UFget/mat                   default download directory (can be changed)
    UFget/mat/UF_Index.mat      index to the UF sparse matrix collection

To install the package, just add the path containing the UFget directory
to your MATLAB path.  Type "pathtool" in MATLAB for more details.

For a simple example of use, type this command in MATLAB:

    UFget_example

The MATLAB statement

    Problem = UFget ('HB/arc130')

(for example), will download a sparse matrix called HB/arc130 (a laser
simulation problem) and load it into MATLAB.  You don't need to use your
web browser to load the matrix.  The statement

    Problem = UFget (6)

will also load same the HB/arc130 matrix.  Each matrix in the collection has
a unique identifier, in the range 1 to the number of matrices.  As new
matrices are added, the id's of existing matrices will not change.

To view an index of the entire collection, just type

    UFget

in MATLAB.  To modify your download directory, edit the UFget_defaults.m file.

To open a URL of the entire collection, just type

    UFweb

To open the URL of a group of matrices in the collection:

    UFweb ('HB')

To open the web page for one matrix, use either of these formats:

    UFweb ('HB/arc130')
    UFweb (6)

To download a new index, to get access to new matrices:

    UFget ('refresh')

(by default, using UFget downloads the index every 90 days anyway).

For more information on how the index entries were created, see
http://www.cise.ufl.edu/research/sparse/SuiteSparse.
