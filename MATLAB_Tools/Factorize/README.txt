 FACTORIZE:  an object-oriented method for solving linear systems and least
  squares problems.  The method provides an efficient way of computing
  mathematical expressions involving the inverse, without actually
  computing the inverse.  For example, S=A-B*inverse(D)*C computes the
  Schur complement by computing S=A-B*(D\C) instead.
 
    factorize  - an object-oriented method for solving linear systems.
    factorize1 - a simple and easy-to-read version of factorize.
    inverse    - factorized representation of inv(A).
    fdemo      - a demo of how to use the FACTORIZE object
 
  Example
    fdemo       % run the demo in the Factorize/Demo directory.
 
  "Don't let that INV go past your eyes; to solve that system, FACTORIZE!"

  Installation and testing:
 
  To install this package, type "pathtool" in the MATLAB command window.
  Add the directory that contains this Factorize/Contents.m file to the
  path.  Save the path for future use.  Alternatively, type these commands
  while in this directory:
 
    addpath(pwd)
    savepath
 
  If you do not have the proper file permissions to save your path, create
  a startup.m file that includes the command "addpath(here)" where "here"
  is the directory containing this file.  Type "help startup" for more
  information.
 
  The Test/ subdirectory contains functions that test this package.
 
  The html/ subdirectory contains a document that illustrates how to use
  the package (the output of fdemo).

  Copyright 2009, Timothy A. Davis, University of Florida.  May 19, 2009.
