#ifndef INCLUDED_constants
#define INCLUDED_constants

#define EFFICIENT_TRANSPOSE
/* #define WEIGHT_MATRIX_PADDING */

#define MATRIX_FILENAME "dat_files/weight_matrix.dat"

#define AGG_WARP_FILT
/* #define BIT_CODING */
#define MULTI_WEIGHT_MATRIX

#define DTYPE float
//Define USING_DOUBLES only if DTYPE==double
/* #define USING_DOUBLES */

//use dpmm instead of dpvm (which is also implemented, but takes far too long (22us per call))
#define USE_DPMM

/* #define SAVE_WEIGHTS */

#endif
