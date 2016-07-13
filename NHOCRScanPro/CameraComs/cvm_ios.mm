//
//  cvm_ios.m
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "cvm_ios.h"
#import "NHGPUTransformFilter.h"

const cvm_point kWarpGLDestPoints[4] = {
    cvm_create_point(-1, -1), // bottom-left GL -> top-left in image
    cvm_create_point( 1, -1), // bottom-right GL -> top-right in image
    cvm_create_point(-1,  1), // top-left GL -> bottom-left in image
    cvm_create_point( 1,  1), // top-right GL -> bottom-right in image
};

void *mz_create(void) {
    void *filter = (__bridge_retained void *)[[NHGPUTransformFilter alloc] initWithSize:CGSizeMake(kLandscapeSampleWidth, kLandscapeSampleHeight)];
    return filter;
}

// free up any persistent references to OpenGL textures, programs, shaders, etc, as well
// as any wrapping objects.
void mz_destroy(void *mz) {
    NHGPUTransformFilter *filter = (__bridge_transfer NHGPUTransformFilter *)mz; // needed to "release" filter
#pragma unused(filter)
}

// tell the filter to call glFinish() on its context
void mz_prepare_for_backgrounding(void *mz) {
    [(__bridge NHGPUTransformFilter *)mz finish];
}

void ios_gpu_unwarp(IplImage *input, const cvm_point from_points[4], IplImage *output){
    // Create filter if necessary
    NHGPUTransformFilter *filter = [[NHGPUTransformFilter alloc] initWithSize:CGSizeMake(input->width, input->height)];
    
    //void *tempCtx = (__bridge_retained void *)filter;
    
    // Calculate and set perspective matrix, then process the image.
    float perspMat[16];
    llcv_calc_persp_transform(perspMat, 16, false, from_points, kWarpGLDestPoints);
    [filter setPerspectiveMat:perspMat];
    //NSLog(@"在Objective－C＋＋中给OJBC传值了，空的么？:%@",input==NULL?@"是的":@"不是的");
    [filter processIplImage:input dstIplImg:output];
}
