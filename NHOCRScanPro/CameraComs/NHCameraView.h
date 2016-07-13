//
//  NHCameraView.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "NHGuideLayer.h"
#import "NHVideoFrame.h"
#import "NHVideoStream.h"
#import "NHVideoStreamDelegate.h"

@class CardIOVideoFrame;
@class CardIOVideoStream;

@interface NHCameraView : UIView<NHVideoStreamDelegate, NHGuideLayerDelegate>

- (id)initWithFrame:(CGRect)frame delegate:(id<NHVideoStreamDelegate>)delegate;

//- (void)updateLightButtonState;

- (void)willAppear;
- (void)willDisappear;

- (void)startVideoStreamSession;
- (void)stopVideoStreamSession;

- (CGRect)guideFrame;

// CGRect for the actual camera preview area within the cameraView
- (CGRect)cameraPreviewFrame;

@property(nonatomic, weak, readwrite) id<NHVideoStreamDelegate> delegate;
@property(nonatomic, strong, readwrite) UIFont *instructionsFont;
@property(nonatomic, assign, readwrite) BOOL suppressFauxCardLayer;

@end
