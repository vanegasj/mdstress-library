#ifndef MDS_ERROR_H
#define MDS_ERROR_H

#include "mds_common.h"
#include "mds_defines.h"

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define MDS_OK     ( 0)
#define MDS_NOT_OK (~0)

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////
#ifndef __FUNCTION__
#define __FUNCTION__ __func__
#endif
#define checkError(a, b) __checkError(a, b, __FILE__, __FUNCTION__, __LINE__)

namespace mds {
    ////////////////////////////////////////////////////////////////////////////////
    // Typedefs
    ////////////////////////////////////////////////////////////////////////////////
    typedef unsigned int mds_error;

    ////////////////////////////////////////////////////////////////////////////////
    // Header-only Routines
    ////////////////////////////////////////////////////////////////////////////////
    /*******************************************************************************
     * __checkError
     *
     * This function is used with a macro to associate file names, function names
     * and file lines with error conditions. Used in conjunction with macro
     * checkError(a,b).
     *
     * PARAMS: bool condition - if true, an error has been encountered
     *         char *details - a detailed string to print on error
     *         char *file - the file name, supplied by macro
     *         char *function - the function name, supplied by macro
     *         int line - the file line, supplied by macro
     *
     * RETURN: int - MDS_OK if error free, otherwise MDS_NOT_OK
     ******************************************************************************/
    int __checkError(
            bool condition,
            const char * details,
            const char * file,
            const char * function,
            int line)
    {
        int mdsStatus = MDS_OK;

        if (condition) {
#if defined(DEBUG) || defined(_DEBUG)
            fprintf(stderr, "Error: %s (%s in %s, line %d)\n", details, function, file, line);
            assert(false == condition);
#else
            fprintf(stderr, "%s\n", details);
#endif
            mdsStatus = MDS_NOT_OK;
        }

        return mdsStatus;
    }
};

#endif//MDS_ERROR_H
