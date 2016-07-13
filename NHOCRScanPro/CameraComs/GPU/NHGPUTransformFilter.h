//
//  NHGPUTransformFilter.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//
#pragma once
#import "NHGPUFilter.h"

@interface NHGPUTransformFilter : NHGPUFilter{
    GLfloat orthographicMatrix[16];
    GLuint _transformMatrixUniform, _orthographicMatrixUniform;
}

// Initialize with the size of the input texture
- (id)initWithSize:(CGSize)size;

// Set the perspective matrix for this filter.
// Array of floats in column-major format.
- (void)setPerspectiveMat:(float *)matrix;

@end
