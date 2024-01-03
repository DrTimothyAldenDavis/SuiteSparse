/* ========================================================================== */
/* === Source/Mongoose_Logger.cpp =========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library, Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * SPDX-License-Identifier: GPL-3.0-only
 * -------------------------------------------------------------------------- */

/**
 * Centralized debug and timing manager
 *
 * For debug and timing information to be displayed via stdout. This system
 * allows this information to be displayed (or not) without recompilation.
 * Timing inforation for different *portions of the library are also managed
 * here with a tic/toc pattern.
 */

#include "Mongoose_Logger.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include <iomanip>

namespace Mongoose
{

int Logger::debugLevel = None;
bool Logger::timingOn  = false;
double Logger::clocks[6];
float Logger::times[6];

/**
 * Start a timer for a given type/part of the code.
 *
 * Given a timingType (MatchingTiming, CoarseningTiming, RefinementTiming,
 * FMTiming, QPTiming, or IOTiming), a clock is started for that computation.
 * The general structure is to call tic(IOTiming) at the beginning of an I/O
 * operation, then call toc(IOTiming) at the end of the I/O operation.
 *
 * Note that problems can occur and timing results may be inaccurate if a tic
 * is followed by another tic (or a toc is followed by another toc).
 *
 * @param timingType The portion of the library being timed (MatchingTiming,
 *   CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 */
void Logger::tic(TimingType timingType)
{
    if (timingOn)
    {
        clocks[timingType] = SUITESPARSE_TIME;
    }
}

/**
 * Stop a timer for a given type/part of the code.
 *
 * Given a timingType (MatchingTiming, CoarseningTiming, RefinementTiming,
 * FMTiming, QPTiming, or IOTiming), a clock is stopped for that computation.
 * The general structure is to call tic(IOTiming) at the beginning of an I/O
 * operation, then call toc(IOTiming) at the end of the I/O operation.
 *
 * Note that problems can occur and timing results may be inaccurate if a tic
 * is followed by another tic (or a toc is followed by another toc).
 *
 * @param timingType The portion of the library being timed (MatchingTiming,
 *   CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 */
void Logger::toc(TimingType timingType)
{
    if (timingOn)
    {
        times[timingType]
            += (float) (SUITESPARSE_TIME - clocks[timingType]) ;
    }
}

/**
 * Get the time recorded for a given timing type.
 *
 * Retreive the total clock time for a given timing type (MatchingTiming,
 * CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 *
 * @param timingType The portion of the library being timed (MatchingTiming,
 *   CoarseningTiming, RefinementTiming, FMTiming, QPTiming, or IOTiming).
 */
float Logger::getTime(TimingType timingType)
{
    return times[timingType];
}

int Logger::getDebugLevel()
{
    return debugLevel;
}

void Logger::setDebugLevel(int debugType)
{
    debugLevel = debugType;
}

void Logger::setTimingFlag(bool tFlag)
{
    timingOn = tFlag;
}

void Logger::printTimingInfo()
{
    std::cout << " Matching:   " << std::setprecision(4)
              << times[MatchingTiming] << "s\n";
    std::cout << " Coarsening: " << std::setprecision(4)
              << times[CoarseningTiming] << "s\n";
    std::cout << " Refinement: " << std::setprecision(4)
              << times[RefinementTiming] << "s\n";
    std::cout << " FM:         " << std::setprecision(4) << times[FMTiming]
              << "s\n";
    std::cout << " QP:         " << std::setprecision(4) << times[QPTiming]
              << "s\n";
    std::cout << " IO:         " << std::setprecision(4) << times[IOTiming]
              << "s\n";
}

} // end namespace Mongoose
