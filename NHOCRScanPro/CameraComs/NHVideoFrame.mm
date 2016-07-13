//
//  NHVideoFrame.m
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHVideoFrame.h"
#import "NHConstants.h"
#import "NHCVMBridge.h"
#import "NHIplImage.h"
#import <opencv2/core/core_c.h>

#define kMinLuma 100
#define kMaxLuma 200
#define kMinFallbackFocusScore 6
#define kMinNonSuckyFocusScore 3

// 以下定义宏为从有效身份证区域截取 姓名&地址 和 号码 子区域 用
#define NAMEROI_WIDTH 0.48
#define NAMEROI_HEIGH 0.75
#define NAMEROI_XPOS  0.175
#define NAMEROI_YPOS  0.05

#define ID_WIDTH 0.61
#define ID_HEIGH 0.15
#define ID_XPOS  0.29
#define ID_YPOS  0.77

#define BANK_WIDTH 0.90
#define BANK_HEIGH 0.13
#define BANK_XPOS  0.05
#define BANK_YPOS  0.55

// 以下定义宏为从 姓名&地址 和 号码 子区域 进行二值分割用

#define NAME_THRE       15
#define ID_THRE         30
#define BANK_THRE       10

using namespace opencvm;
bool getState();
float cvm_focus_score(IplImage *image, bool use_full_image);
float cvm_brightness_score(IplImage *image, bool use_full_image);
bool cvm_detect_edges(IplImage *y_sample, IplImage *cb_sample, IplImage *cr_sample,FrameOrientation orientation, cvm_edges *found_edges, cvm_corner_points *corner_points);
void cvm_transform_card( IplImage *sample, cvm_corner_points corner_points, FrameOrientation orientation, bool upsample, IplImage **transformed);
IplImage *segementImg(IplImage *src, CardType type,float x, float y, float w,float h, float t);

@interface NHVideoFrame ()

@property (nonatomic, assign, readwrite) CMSampleBufferRef buffer;
@property (nonatomic, assign, readwrite) UIInterfaceOrientation orientation;
@property (nonatomic, assign, readwrite) cvm_edges found_edges;
@property (nonatomic, assign, readwrite) cvm_corner_points corner_points;
@property (assign) Cvm m_cvm;

- (void)detectCardInSamples;
- (void)detectCardInSamplesWithFlip:(BOOL)shouldFlip;
- (void)transformCbCrWithFrameOrientation:(FrameOrientation)frameOrientation;

@end

@implementation NHVideoFrame

- (id)initWithSampleBuffer:(CMSampleBufferRef)sampleBuffer interfaceOrientation:(UIInterfaceOrientation)currentOrientation {
    if((self = [super init])) {
        _buffer = sampleBuffer;
        _orientation = currentOrientation; // not using setters/getters, for performance
        //_cvm = NULL;  // use NULL b/c non-object pointer
        
    }
    return self;
}

- (void)process {
    BOOL performAllProcessing = NO;
    
    cvSetErrMode(CV_ErrModeParent);
    
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(self.buffer);
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    self.ySample = [NHIplImage imageFromYCbCrBuffer:imageBuffer plane:Y_PLANE];
    
    BOOL useFullImageForFocusScore = NO;
    //useFullImageForFocusScore = (self.detectionMode == CardIODetectionModeCardImageOnly); // when detecting, rely more on focus than on contents
    
    self.focusScore = _m_cvm.cvm_focus_score(self.ySample.image, useFullImageForFocusScore);
    self.focusOk = self.focusScore > kMinFallbackFocusScore;
    self.focusSucks = self.focusScore < kMinNonSuckyFocusScore;
    
    if (self.calculateBrightness) {
        self.brightnessScore = _m_cvm.cvm_brightness_score(self.ySample.image, self.torchIsOn);
        self.brightnessLow = self.brightnessScore < kMinLuma;
        self.brightnessHigh = self.brightnessScore > kMaxLuma;
    }
    
//    if(self.detectionMode == CardIODetectionModeCardImageOnly) {
//        performAllProcessing = YES;
//    }
    
    if(self.focusOk || performAllProcessing) {
        NHIplImage *brSample = [NHIplImage imageFromYCbCrBuffer:imageBuffer plane:CBCR_PLANE];
        
        NSArray *bAndRSamples = [brSample split];
        self.cbSample = bAndRSamples[0];
        self.crSample = bAndRSamples[1];
        
        [self detectCardInSamples];
    }
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    
#if CARDIO_DEBUG
    //  self.debugString = [NSString stringWithFormat:@"Focus: %5.1f", focusScore];
#endif
}

- (void)detectCardInSamples {
    [self detectCardInSamplesWithFlip:NO];
}

- (void)detectCardInSamplesWithFlip:(BOOL)shouldFlip {
    self.flipped = shouldFlip;
    
    FrameOrientation frameOrientation = frameOrientationWithInterfaceOrientation(self.orientation);
    if (self.flipped) {
        frameOrientation = cvm_opposite_orientation(frameOrientation);
    }
    
    bool foundCard = _m_cvm.cvm_detect_edges(self.ySample.image, self.cbSample.image, self.crSample.image,frameOrientation, &_found_edges, &_corner_points);
    
    self.foundTopEdge = (BOOL)self.found_edges.top.found;
    self.foundBottomEdge = (BOOL)self.found_edges.bottom.found;
    self.foundLeftEdge = (BOOL)self.found_edges.left.found;
    self.foundRightEdge = (BOOL)self.found_edges.right.found;
    
    if (foundCard) {
        IplImage *foundCardY = 0;
        _m_cvm.cvm_transform_card( self.ySample.image, self.corner_points, frameOrientation, false, &foundCardY);
        self.cardY = [NHIplImage imageWithIplImage:foundCardY];
        [self transformCbCrWithFrameOrientation:frameOrientation];
        
        CardType type = CardTypeID;
        bool cid = type == CardTypeID;
        IplImage *temp = _m_cvm.segementImg(self.cardY.image,type, cid?ID_XPOS:BANK_XPOS, cid?ID_YPOS:BANK_YPOS, cid?ID_WIDTH:BANK_WIDTH, cid?ID_HEIGH:BANK_HEIGH, cid?ID_THRE:BANK_THRE);
        self.retImg = [NHIplImage imageWithIplImage:temp];
        
        //printf("width:%d---height:%d\n",temp->width,temp->height);
        
        BOOL scan = false;
        if (scan) {
            
        }else{
            // we're not scanning, so the transformed cb/cr channels might be needed at any time
            //[self transformCbCrWithFrameOrientation:frameOrientation];
        }
    }
}

- (void)transformCbCrWithFrameOrientation:(FrameOrientation)frameOrientation {
    // It's safe to calculate cardCb and cardCr if we've already calculated cardY, since they share
    // the same prerequisites.
    if(self.cardY) {
        IplImage *foundCardCb = NULL;
        _m_cvm.cvm_transform_card( self.cbSample.image, self.corner_points, frameOrientation, true, &foundCardCb);
        self.cardCb = [NHIplImage imageWithIplImage:foundCardCb];
        
        IplImage *foundCardCr = NULL;
        _m_cvm.cvm_transform_card(self.crSample.image, self.corner_points, frameOrientation, true, &foundCardCr);
        self.cardCr = [NHIplImage imageWithIplImage:foundCardCr];
    }
}

- (BOOL)foundAllEdges {
    return self.foundTopEdge && self.foundBottomEdge && self.foundLeftEdge && self.foundRightEdge;
}

- (uint)numEdgesFound {
    return (uint) self.foundTopEdge + (uint) self.foundBottomEdge + (uint) self.foundLeftEdge + (uint) self.foundRightEdge;
}

- (UIImage *)imageWithGrayscale:(BOOL)grayscale {
    return grayscale ? [self.cardY UIImage] : [[NHIplImage rgbImageWithY:self.cardY cb:self.cardCb cr:self.cardCr] UIImage];
}

@end
