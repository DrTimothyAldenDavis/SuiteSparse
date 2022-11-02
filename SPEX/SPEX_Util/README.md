This folder contains utility functions for the SPEX software package.
These functions are shared amongst all other SPEX packages.

Key subroutines within this folder:
    
    -All SPEX GMP/MPFR wrapper functions
    -All SPEX matrix functions
    -All SPEX memory handling functions
    -General Ax = b sanity check function

Other functions that may be useful can be added in the future.

Note that all of these functions are test covered by the respective 
packages that use them. For example, SPEX Left LU performs test coverage
on every used utility function.
