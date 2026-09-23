#-------------------------------------------------------------------------------
# SPEX/Python/spex_python_demo.py: demo of 3 backslash functions with different input 
#                           matrices
#-------------------------------------------------------------------------------

# SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------

# SPEX is a package for solving sparse linear systems of equations
# with a roundoff-free integer-preserving method.  The result is
# always exact, unless the matrix A is perfectly singular.

# Import SPEX
import SPEXpy as SPEX
from SPEXpy import Options

# Import scientific computing
import numpy as np
from numpy.random import default_rng
from scipy.sparse import csc_matrix
from scipy.sparse import random
from scipy import stats


##--------------------------------------------------------------------------
## Cholesky
##--------------------------------------------------------------------------

# Create A and B
print("Demoing the Cholesky interface")
row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
col = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
data = np.array([4, 12, -16, 12, 37, -43, -16, -43, 98],dtype=np.float64)
A1=csc_matrix((data, (row, col)), shape=(3, 3))
b1=np.ones(3,dtype=np.float64)

# Solve
x=SPEX.cholesky_backslash(A1,b1)
print("\nThe Cholesky solution is:\n")
print(x)


##--------------------------------------------------------------------------
## LU
##--------------------------------------------------------------------------

# Generate a random sparse matrix A and populate b
print("Demoing the LU interface")
n=10
rng = default_rng()
rvs = stats.poisson(25, loc=10).rvs
S = random(n, n, density=0.7, random_state=rng, data_rvs=rvs)
S2=S+np.eye(n)
A2=csc_matrix(S2)
b2=np.ones(n,dtype=np.float64)

# Solve
options=Options("string")
x=SPEX.lu_backslash(A2,b2,options)
print("\nThe LU solution is:\n")
print(x)

##--------------------------------------------------------------------------
## QR
##--------------------------------------------------------------------------

# Generate a random sparse matrix A and populate b
print("Demoing the QR solution")
n=7
m=10
rng = default_rng()
rvs = stats.poisson(25, loc=10).rvs
S = random(m, n, density=0.7, random_state=rng, data_rvs=rvs)
S2=S+np.eye(m, n)
A3=csc_matrix(S2)
b3=np.ones(m,dtype=np.float64)

# Solve
options=Options("string")
x=SPEX.qr_backslash(A3,b3,options)
print("\nThe QR solution is:\n")
print(x)

print("\nDemoing rank")
print("\nThe rank of square matrix 1 is:")
r = SPEX.rank(A1)
print(r)

print("\nThe rank of square matrix 2 is:")
r = SPEX.rank(A2)
print(r)

print("\nThe rank of rectangular matrix is:")
r = SPEX.rank(A3)
print(r)


##--------------------------------------------------------------------------
## Backslash
##--------------------------------------------------------------------------

print("Demoing SPEX backslash")
# Use the previous matrices

# Solve
x=SPEX.backslash(A1,b1)
print("\nLDL good")
x=SPEX.backslash(A2,b2)
print("\nLU good")
x=SPEX.backslash(A3,b3)
print("\nQR good")
print("All set, you're ready to go!")
#SPEX.backslash always returns the output as float64
