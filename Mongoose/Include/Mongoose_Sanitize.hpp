/* ========================================================================== */
/* === Include/Mongoose_Sanitize.hpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#pragma once

#include "Mongoose_CSparse.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

cs *sanitizeMatrix(cs *compressed_A, bool symmetricTriangular,
                   bool makeEdgeWeightsBinary);
void removeDiagonal(cs *A);
// Requires A to be a triangular matrix with no diagonal.
cs *mirrorTriangular(cs *A);

} // end namespace Mongoose
