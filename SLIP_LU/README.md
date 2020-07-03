SLIP_LU is software package used to solve a sparse systems of linear equations
exactly using the Sparse Left-looking Integer-Preserving LU factorization.

*********SLIPLU*********
Purpose: Exactly solve a sparse system of linear equations using a given input
         matrix and right hand side vector file. This code can output the final
         solution to a user specified output file in either double precision or
         full precision rational numbers. If you intend to use SLIP LU within
         another program, refer to examples for help with this.

./SLIPLU followed by the listed args:

help. e.g., ./SLIPLU help, which indicates SLIPLU to print to guideline
for using this function.

f (or file) Filename. e.g., ./SLIPLU f MATRIX_NAME RHS_NAME, which indicates
SLIPLU will read matrix from MATRIX_NAME and right hand side from RHS_NAME.
For this demo, the matrix is stored in a triplet format. Refer to
SLIP_LU/ExampleMats for examples.

p (or piv) Pivot_param. e.g., ./SLIPLU p 0, which indicates SLIPLU will use
smallest pivot for pivot scheme. Other available options are listed
as follows:
       0: Smallest pivot
       1: Diagonal pivoting
       2: First nonzero per column chosen as pivot
       3: Diagonal pivoting with tolerance for smallest pivot, Default
       4: Diagonal pivoting with tolerance for largest pivot
       5: Largest pivot

q (or col) Column_order_param. e.g., ./SLIPLU q 0, which indicates SLIPLU
will use COLAMD for column ordering. Other available options are:
       0: None: Not recommended for sparse matrices
       1: COLAMD: Default
       2: AMD

t (or tol) tolerance_param. e.g., ./SLIPLU t 1e-10, which indicates SLIPLU
will use 1e-10 as the tolerance for pivot scheme 3 and 4 mentioned above.
Therefore, it is only necessary if pivot scheme 3 or 4 is used.

o (or out). e.g., SLIPLU o 1, which indicates SLIPLU will output the
errors and warnings during the process. Other available options are:
       0: print nothing
       1: just errors and warnings: Default
       2: terse, with basic stats from COLAMD/AMD and SLIP and solution

If none of the above args is given, they are set to the following default:

  mat_name = "../ExampleMats/10teams_mat.txt"
  rhs_name = "../ExampleMats/10teams_v.txt"
  p = 3, 
  q = 1, i.e., using COLAMD
  t = 1,


*********example*********
Purpose: Demonstrate the simple interface of SLIP LU for a randomly generated
         matrix

*********example2*********
Purpose: Demonstrate the simple interface of SLIP LU for a matrix to be read in

