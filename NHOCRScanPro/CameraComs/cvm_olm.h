//
//  cvm_olm.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#ifndef __NHOpenCVPro__cvm_olm__
#define __NHOpenCVPro__cvm_olm__

//#include "cvm_prep.h"
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "cvm_constants.h"
#include "eigen.h"

using namespace std;
using namespace cv;

/******* Types *******/

typedef struct {
    void *mz;
}cvm_context;

typedef uint8_t FrameOrientation;
enum {
    FrameOrientationPortrait = 1, // == UIInterfaceOrientationPortrait
    FrameOrientationPortraitUpsideDown = 2, // == UIInterfaceOrientationPortraitUpsideDown
    FrameOrientationLandscapeRight = 3, // == UIInterfaceOrientationLandscapeRight
    FrameOrientationLandscapeLeft = 4 // == UIInterfaceOrientationLandscapeLeft
} ;


typedef struct {
    float x;
    float y;
} cvm_point;

typedef struct {
    float x;
    float y;
    float w;
    float h;
} cvm_rect;

typedef struct {
    cvm_point top_left;
    cvm_point bottom_left;
    cvm_point top_right;
    cvm_point bottom_right;
} cvm_corner_points;

typedef struct {
    float rho;
    float theta;
} ParametricLine;

typedef struct {
    int found; // bool indicating whether this edge was detected; if 0, the other values in this struct may contain garbage
    ParametricLine location;
} cvm_found_edge;

typedef struct {
    cvm_found_edge top;
    cvm_found_edge left;
    cvm_found_edge bottom;
    cvm_found_edge right;
} cvm_edges;

cvm_point cvm_create_point(float x, float y);

cvm_rect cvm_create_rect(float x, float y, float w, float h);

void cvm_rect_get_points(cvm_rect rect, cvm_point points[4]);

cvm_context *cvm_context_create();
void cvm_context_destroy(cvm_context *cvm);
void cvm_prepare_for_backgrounding(cvm_context *cvm);

cvm_rect cvm_guide_frame(FrameOrientation orientation, float preview_width, float preview_height);

FrameOrientation cvm_opposite_orientation(FrameOrientation orientation);

// unwarps input image, interpolating image such that src_points map to dst_rect coordinates.
// Image is written to output IplImage.
void llcv_unwarp(IplImage *input, const cvm_point src_points[4], const cvm_rect dst_rect, IplImage *output);

// Solves and writes perpsective matrix to the matrixData buffer.
// If matrixDataSize >= 16, uses a 4x4 matrix. Otherwise a 3x3.
// Specifying rowMajor true writes to the buffer in row major format.
void llcv_calc_persp_transform(float *matrixData, int matrixDataSize, bool rowMajor, const cvm_point sourcePoints[], const cvm_point destPoints[]);

void* llcv_get_data_origin(IplImage *image);

void cvm_deinterleave_RGBA_to_R(uint8_t *source, uint8_t *dest, int size);

void cvm_YCbCr_to_RGB(IplImage *y, IplImage *cb, IplImage *cr, IplImage **rgb);

int otsu(IplImage *image);

int ThresholdOtsu(cv::Mat mat);

int sortRect(const vector<CvRect>& vecRect, vector<CvRect>& out);

#endif /* defined(__NHOpenCVPro__cvm_olm__) */
