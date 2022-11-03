//------------------------------------------------------------------------------
// Mongoose/Tests/Mongoose_Test_EdgeSeparator_exe.cpp
//------------------------------------------------------------------------------

// Mongoose Graph Partitioning Library, Copyright (C) 2017-2018,
// Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
// Mongoose is licensed under Version 3 of the GNU General Public License.
// Mongoose is also available under other licenses; contact authors for details.
// SPDX-License-Identifier: GPL-3.0-only

//------------------------------------------------------------------------------

#include "Mongoose_Test.hpp"

using namespace Mongoose;

#undef LOG_ERROR
#undef LOG_WARN
#undef LOG_INFO
#undef LOG_TEST
#define LOG_ERROR 1
#define LOG_WARN 1
#define LOG_INFO 0
#define LOG_TEST 1

int main(int argn, char** argv)
{
    SuiteSparse_start();

    if (argn < 2 || argn > 3)
    {
        // Wrong number of arguments - return error
        SuiteSparse_finish();
        return EXIT_FAILURE;
    }

    double targetSplit;

    if (argn == 2)
    {
        // Default split of 0.5
        targetSplit = 0.5;
    }
    else
    {
        targetSplit = atof(argv[2]);
    }

    // Read in input file name
    std::string inputFile = std::string(argv[1]);

    // Set Logger to report all messages
    Logger::setDebugLevel(All);

    // Turn timing information on
    Logger::setTimingFlag(true);

    // Run the Edge Separator test
    int status = runEdgeSeparatorTest(inputFile, targetSplit);

    SuiteSparse_finish();

    return status;
}
