//
//  NHVideoFrame.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreMedia/CoreMedia.h>
#import "cvm_olm.h"
#import "cvm_prep.h"
#import <opencv2/imgproc/imgproc_c.h>

typedef enum {
    FrameEncodingColorPNG = 1,
    FrameEncodingGrayPNG = 2
} FrameEncoding;

@class NHIplImage;
@interface NHVideoFrame : NSObject

- (id)initWithSampleBuffer:(CMSampleBufferRef)sampleBuffer interfaceOrientation:(UIInterfaceOrientation)currentOrientation;

- (void)process;
- (BOOL)foundAllEdges;
- (uint)numEdgesFound;
- (UIImage *)imageWithGrayscale:(BOOL)grayscale;

@property(nonatomic, assign, readwrite) float focusScore;
@property(nonatomic, assign, readwrite) BOOL focusOk;
@property(nonatomic, assign, readwrite) BOOL focusSucks;
@property(nonatomic, assign, readwrite) float brightnessScore;
@property(nonatomic, assign, readwrite) BOOL brightnessLow;
@property(nonatomic, assign, readwrite) BOOL brightnessHigh;
@property(nonatomic, assign, readwrite) BOOL foundTopEdge;
@property(nonatomic, assign, readwrite) BOOL foundBottomEdge;
@property(nonatomic, assign, readwrite) BOOL foundLeftEdge;
@property(nonatomic, assign, readwrite) BOOL foundRightEdge;
@property(nonatomic, assign, readwrite) BOOL flipped;
@property(nonatomic, assign, readwrite) BOOL scanExpiry;
@property(nonatomic, assign, readwrite) NSInteger isoSpeed;
@property(nonatomic, assign, readwrite) float shutterSpeed;
@property(nonatomic, strong, readwrite) NHIplImage *ySample;
@property(nonatomic, strong, readwrite) NHIplImage *cbSample;
@property(nonatomic, strong, readwrite) NHIplImage *crSample;
@property(nonatomic, strong, readwrite) NHIplImage *cardY;
@property(nonatomic, strong, readwrite) NHIplImage *cardCb;
@property(nonatomic, strong, readwrite) NHIplImage *cardCr;
@property(nonatomic, strong, readwrite) NHIplImage *retImg;
@property(nonatomic, assign) vector<Mat>mats;
@property(nonatomic, assign) vector<IplImage*>imgs;
@property(assign) cv::Mat matimg;
//@property(nonatomic, strong, readwrite) CardIOCardScanner *scanner;
//@property(nonatomic, strong, readwrite) CardIOReadCardInfo *cardInfo; // Will be nil unless frame processing completes with a successful scan
@property(assign) cvm_context *cvm;
@property(nonatomic, assign, readwrite) BOOL calculateBrightness;
@property(nonatomic, assign, readwrite) BOOL torchIsOn;


@end
