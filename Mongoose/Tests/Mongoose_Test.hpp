//------------------------------------------------------------------------------
// Mongoose/Tests/Mongoose_Test.hpp
//------------------------------------------------------------------------------

// Mongoose Graph Partitioning Library, Copyright (C) 2017-2018,
// Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
// Mongoose is licensed under Version 3 of the GNU General Public License.
// Mongoose is also available under other licenses; contact authors for details.
// SPDX-License-Identifier: GPL-3.0-only

//------------------------------------------------------------------------------

#ifndef Mongoose_Test_hpp
#define Mongoose_Test_hpp

#include <climits>
#include <cstdlib>
#include <cassert>

/* Dependencies */
#include "stddef.h"
#include "stdlib.h"
#include "math.h"

/* Memory Management */
#include "SuiteSparse_config.h"

#include <string>

int runIOTest(const std::string &inputFile, bool validGraph);
int runMemoryTest(const std::string &inputFile);
int runTimingTest(const std::string &inputFile);
int runEdgeSeparatorTest(const std::string &inputFile, const double targetSplit);
int runPerformanceTest(const std::string &inputFile, const std::string &outputFile);

// Currently unused
int runReferenceTest(const std::string &inputFile);

#include "Mongoose_Logger.hpp"

#endif
