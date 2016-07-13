//
//  NHVideoStream.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import "NHVideoStreamDelegate.h"
#import "cvm_scan.h"

using namespace opencv_scan;

@interface NHVideoStream : NSObject

- (void)willAppear;
- (void)willDisappear;

- (BOOL)hasTorch;
- (BOOL)canSetTorchLevel;
- (BOOL)torchIsOn;
- (BOOL)setTorchOn:(BOOL)torchShouldBeOn; // returns success value
- (BOOL)hasAutofocus;

- (void)refocus;

- (void)startSession;
- (void)stopSession;

@property(nonatomic, assign, readonly) BOOL running;
@property(nonatomic, weak, readwrite) id<NHVideoStreamDelegate> delegate;
@property(nonatomic, strong, readonly) AVCaptureVideoPreviewLayer *previewLayer;
@property(assign)CvScan m_scanner;

@end
