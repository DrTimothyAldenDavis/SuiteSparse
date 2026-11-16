SPEX QR is a software package used to solve linear systems exactly using the Sparse QR factorization.
It applies a thin QR factorization to either A or A^T depending on the structure of A and
the dimension of the right hand side vector.

SPEX QR can also be utilized to compute the exact rank of a given matrix using a rank revealing QR
factorization.
Importantly, however, if A is square, a user is highly advised to utilize the SPEX_lu_rank or
SPEX_rank functions instead of SPEX_qr_rank because they will be significantly faster and less memory
intensive.
