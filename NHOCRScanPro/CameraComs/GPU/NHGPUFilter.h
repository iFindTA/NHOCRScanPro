//
//  NHGPUFilter.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHGPUShaders.h"
//#import <Foundation/Foundation.h>
#import "NHGPURender.h"
#import <OpenGLES/ES3/gl.h>

@interface NHGPUFilter : NSObject{
    NHGPURender *_gpuRenderer;
}

@property (nonatomic, assign, readonly) CGSize size;

- (id)initWithSize:(CGSize)size vertexShaderSrc:(NSString *)vertexShaderSrc fragmentShaderSrc:(NSString *)fragmentShaderSrc;

// Called upon app backgrounding, to call glFinish()
- (void)finish;

// Returns a new UIImage of the given size
- (UIImage *)processUIImage:(UIImage *)srcUIImage toSize:(const CGSize)size;

- (UIImage *)processIplToUIImage:(IplImage *)srcImg toSize:(const CGSize)size;

// Sets the imageData of the given dstImg IplImage.
- (void)processIplImage:(IplImage *)srcImg dstIplImg:(IplImage *)dstImg;

// Helper class methods
+ (void)loadIdentityMatrix:(GLfloat *)matrix size:(GLuint)size;
+ (void)loadOrthoMatrix:(GLfloat *)matrix left:(GLfloat)left right:(GLfloat)right bottom:(GLfloat)bottom top:(GLfloat)top near:(GLfloat)near far:(GLfloat)far;
+ (NSString *)matrixAsString:(GLfloat *)matrix size:(GLuint)size rowMajor:(BOOL)rowMajor;

@end
