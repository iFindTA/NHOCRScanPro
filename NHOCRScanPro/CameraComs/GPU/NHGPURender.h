//
//  NHGPURender.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//
#pragma once
#ifdef __OBJC__
#import <UIKit/UIKit.h>
#endif
#import "NHGPUShaders.h"
#import <OpenGLES/ES3/gl.h>
#import <opencv2/imgproc/imgproc_c.h>

@interface NHGPURender : NSObject{
    // Handles
    GLuint _programHandle;
    
    // Inputs
    GLuint _inputTexture, _positionSlot, _texCoordSlot, _textureUniform;
}

@property (nonatomic, assign, readonly) CGSize size;

- (id)initWithSize:(CGSize)size vertexShaderSrc:(NSString *)vertexShaderSrc fragmentShaderSrc:(NSString *)vertexShaderSrc;
- (void)finish;

- (void)prepareForUse;
- (GLuint) uniformIndex:(NSString *)uniformName;

// Creates a UIImage from the currently rendered framebuffer.
// Resulting image is 32bit RGBA
- (UIImage *)captureUIImageOfSize:(CGSize)size;

- (void)renderIplImage:(IplImage *)inputImage toSize:(CGSize)targetSize;

- (void)withContextDo:(void (^)(void))successBlock;

- (void)renderUIImage:(UIImage *)inputImage toSize:(CGSize)targetSize;

// Creates an IplImage from the currently rendered framebuffer.
// Resulting image is 8bit grayscale
- (void)captureIplImage:(IplImage *)dstImg;

@end
