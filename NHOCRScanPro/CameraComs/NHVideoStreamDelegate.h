//
//  NHVideoStreamDelegate.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#ifndef NHOpenCVPro_NHVideoStreamDelegate_h
#define NHOpenCVPro_NHVideoStreamDelegate_h

#import <UIKit/UIKit.h>

@class NHVideoFrame;
@class NHVideoStream;

@protocol NHVideoStreamDelegate<NSObject>

@required

- (void)videoStream:(NHVideoStream *)stream didProcessFrame:(NHVideoFrame *)processedFrame;

@optional

- (BOOL)isSupportedOverlayOrientation:(UIInterfaceOrientation)orientation;
- (UIInterfaceOrientation)defaultSupportedOverlayOrientation;

@end

#endif
