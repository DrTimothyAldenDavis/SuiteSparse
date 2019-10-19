UFget:  a MATLAB interface to the UF sparse matrix collection.
MATLAB 7.0 or later is required.

Date: Nov 30, 2006.
Copyright 2005-2006, Tim Davis, University of Florida.
Authors: Tim Davis and Erich Mirable.
Availability: http://www.cise.ufl.edu/research/sparse/mat/UFget

See http://www.cise.ufl.edu/research/sparse/mat/UFget.tar.gz
for a single archive file with all the files listed below.

    UFget/Contents.m		help for UFget
    UFget/README.txt		this file

    UFget/UFget_defaults.m	default parameter settings for UFget
    UFget/UFget_example.m	demo for UFget
    UFget/UFget_install.m	installs UFget for use in MATLAB
    UFget/UFget_java.class	compiled version of UFget_java.java
    UFget/UFget_java.java	downloads a URL
    UFget/UFget_lookup.m	get the group, name, and id of a matrix
    UFget/UFget.m		primary user interface

    UFget/UFweb.m		opens the URL for a matrix or collection

    UFget/mat			default download directory (can be changed)
    UFget/mat/UF_Index.mat	index to the UF sparse matrix collection

You may also need the Java Development Kit (Java 2 Platform, at
http://java.sun.com/j2se/index.html ) to compile UFget_java.java.

To install the package, type this command in MATLAB:

    UFget_install

For a simple example of use, type this command in MATLAB:

    UFget_example

Once the files are downloaded and installed, the MATLAB statement

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

For more information on how the index entries were created, see
http://www.cise.ufl.edu/research/sparse/SuiteSparse.

The UFget/UFget_java.class was compiled using:

    java version "1.5.0"
    Java(TM) 2 Runtime Environment, Standard Edition (build 1.5.0-b64)
    Java HotSpot(TM) Client VM (build 1.5.0-b64, mixed mode)
