#ifndef LAYERDEF
#define LAYERDEF

#define FDIM 3
#define FDIM_P 1
#define FILTER_MAX 1024

/*******************************************************************************
* Defines - Layer 0 Standard Convolution - Stride 2                            *
*******************************************************************************/

#define HEIGHT_0 224
#define WIDTH_0 224
#define IP_FM_0 3
#define OP_FM_0 32


/*******************************************************************************
* Defines - Layer 1 Depthwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_1 112
#define WIDTH_1 112
#define IP_FM_1 32
#define OP_FM_1 32


/*******************************************************************************
* Defines - Layer 2 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_2 112
#define WIDTH_2 112
#define IP_FM_2 32
#define OP_FM_2 64


/*******************************************************************************
* Defines - Layer 3 Depthwise Convolution - Stride 2                           *
*******************************************************************************/

#define HEIGHT_3 112
#define WIDTH_3 112
#define IP_FM_3 64
#define OP_FM_3 64


/*******************************************************************************
* Defines - Layer 4 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_4 56
#define WIDTH_4 56
#define IP_FM_4 64
#define OP_FM_4 128


/*******************************************************************************
* Defines - Layer 5 Depthwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_5 56
#define WIDTH_5 56
#define IP_FM_5 128
#define OP_FM_5 128


/*******************************************************************************
* Defines - Layer 6 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_6 56
#define WIDTH_6 56
#define IP_FM_6 128
#define OP_FM_6 128


/*******************************************************************************
* Defines - Layer 7 Depthwise Convolution - Stride 2                           *
*******************************************************************************/

#define HEIGHT_7 56
#define WIDTH_7 56
#define IP_FM_7 128
#define OP_FM_7 128


/*******************************************************************************
* Defines - Layer 8 Pointwise Convolution - Stride 1                           *
*******************************************************************************/

#define HEIGHT_8 28
#define WIDTH_8 28
#define IP_FM_8 128
#define OP_FM_8 256


/*******************************************************************************
* Defines - Layer 9 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_9 28
#define WIDTH_9 28
#define IP_FM_9 256
#define OP_FM_9 256


/*******************************************************************************
* Defines - Layer 10 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_10 28
#define WIDTH_10 28
#define IP_FM_10 256
#define OP_FM_10 256


/*******************************************************************************
* Defines - Layer 11 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_11 28
#define WIDTH_11 28
#define IP_FM_11 256
#define OP_FM_11 256


/*******************************************************************************
* Defines - Layer 12 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_12 14
#define WIDTH_12 14
#define IP_FM_12 256
#define OP_FM_12 512


/*******************************************************************************
* Defines - Layer 13 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_13 14
#define WIDTH_13 14
#define IP_FM_13 512
#define OP_FM_13 512


/*******************************************************************************
* Defines - Layer 14 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_14 14
#define WIDTH_14 14
#define IP_FM_14 512
#define OP_FM_14 512


/*******************************************************************************
* Defines - Layer 15 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_15 14
#define WIDTH_15 14
#define IP_FM_15 512
#define OP_FM_15 512


/*******************************************************************************
* Defines - Layer 16 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_16 14
#define WIDTH_16 14
#define IP_FM_16 512
#define OP_FM_16 512


/*******************************************************************************
* Defines - Layer 17 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_17 14
#define WIDTH_17 14
#define IP_FM_17 512
#define OP_FM_17 512

/*******************************************************************************
* Defines - Layer 18 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_18 14
#define WIDTH_18 14
#define IP_FM_18 512
#define OP_FM_18 512


/*******************************************************************************
* Defines - Layer 19 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_19 14
#define WIDTH_19 14
#define IP_FM_19 512
#define OP_FM_19 512


/*******************************************************************************
* Defines - Layer 20 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_20 14
#define WIDTH_20 14
#define IP_FM_20 512
#define OP_FM_20 512


/*******************************************************************************
* Defines - Layer 21 Depthwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_21 14
#define WIDTH_21 14
#define IP_FM_21 512
#define OP_FM_21 512


/*******************************************************************************
* Defines - Layer 22 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_22 14
#define WIDTH_22 14
#define IP_FM_22 512
#define OP_FM_22 512


/*******************************************************************************
* Defines - Layer 23 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_23 14
#define WIDTH_23 14
#define IP_FM_23 512
#define OP_FM_23 512


/*******************************************************************************
* Defines - Layer 24 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_24 7
#define WIDTH_24 7
#define IP_FM_24 512
#define OP_FM_24 1024


/*******************************************************************************
* Defines - Layer 25 Depthwise Convolution - Stride 2                          *
*******************************************************************************/

#define HEIGHT_25 7
#define WIDTH_25 7
#define IP_FM_25 1024
#define OP_FM_25 1024


/*******************************************************************************
* Defines - Layer 26 Pointwise Convolution - Stride 1                          *
*******************************************************************************/

#define HEIGHT_26 7
#define WIDTH_26 7
#define IP_FM_26 1024
#define OP_FM_26 1024


/*******************************************************************************
* Defines - Layer 27 Average Pool - Stride 1                                   *
*******************************************************************************/

#define HEIGHT_27 7
#define WIDTH_27 7
#define IP_FM_27 1024
#define OP_FM_27 1024

/*******************************************************************************
* Defines - Layer 28 Fully connected Layer                                     *
*******************************************************************************/

#define HEIGHT_28 1
#define WIDTH_28 1
#define ELEMENTS 1024
#define CLASSES 40


/*******************************************************************************
* Defines - Layer 29 Softmax                                                   *
*******************************************************************************/

#define HEIGHT_29 1
#define WIDTH_29 1
//#define CLASSES_SOFTMAX 1000
#define CLASSES_SOFTMAX 40
/* 
#define FILTER_SIZE_L5 128
#define FILTER_SIZE_L9 256
#define FILTER_SIZE_L13 512
#define FILTER_SIZE_L25 1024
#define FILTER_SIZE_L29 1000
*/
#endif
