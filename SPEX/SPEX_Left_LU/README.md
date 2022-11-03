
SPEX_Left_LU is software package used to solve a sparse systems of linear equations
exactly using the Sparse Left-looking Integer-Preserving LU factorization.

*********SPEX_Left_LU*********
Purpose: Exactly solve a sparse system of linear equations using a given input
         matrix and right hand side vector file. This code can output the final
         solution to a user specified output file in either double precision or
         full precision rational numbers. If you intend to use SPEX_Left_LU within
         another program, refer to examples for help with this.

./spexlu_demo followed by the listed args:

help. e.g., ./spexlu_demo help, which indicates to print to guideline
for using this function.

f (or file) Filename. e.g., ./spexlu_demo f MATRIX_NAME RHS_NAME, which indicates
SPEX_Left_LU will read matrix from MATRIX_NAME and right hand side from RHS_NAME.
For this demo, the matrix is stored in a triplet format. Refer to
SPEX_Left_LU/ExampleMats for examples.

p (or piv) Pivot_param. e.g., ./spexlu_demo p 0, which indicates SPEX_Left_LU will use
smallest pivot for pivot scheme. Other available options are listed
as follows:
       0: Smallest pivot
       1: Diagonal pivoting
       2: First nonzero per column chosen as pivot
       3: Diagonal pivoting with tolerance for smallest pivot, Default
       4: Diagonal pivoting with tolerance for largest pivot
       5: Largest pivot

q (or col) Column_order_param. e.g., ./spexlu_demo q 0, which indicates SPEX_Left_LU
will use COLAMD for column ordering. Other available options are:
       0: None: Not recommended for sparse matrices
       1: COLAMD: Default
       2: AMD

t (or tol) tolerance_param. e.g., ./spexlu_demo t 1e-10, which indicates SPEX_Left_LU
will use 1e-10 as the tolerance for pivot scheme 3 and 4 mentioned above.
Therefore, it is only necessary if pivot scheme 3 or 4 is used.

o (or out). e.g., ./spexlu_demo o 1, which indicates SPEX_Left_LU will output the
errors and warnings during the process. Other available options are:
       0: print nothing
       1: just errors and warnings: Default
       2: terse, with basic stats from COLAMD/AMD and SPEX and solution

If none of the above args is given, they are set to the following default:

  mat_name = "../ExampleMats/10teams_mat.txt"
  rhs_name = "../ExampleMats/10teams_v.txt"
  p = 3, 
  q = 1, 
  t = 1,


*********example*********
Purpose: Demonstrate the simple interface of SPEX_Left_LU for a randomly generated
         matrix

*********example2*********
Purpose: Demonstrate the simple interface of SPEX_Left_LU for a matrix to be read in

