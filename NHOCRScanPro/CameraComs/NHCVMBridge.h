//
//  NHCVMBridge.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "cvm.h"
#import <Foundation/Foundation.h>

static inline FrameOrientation frameOrientationWithInterfaceOrientation(UIInterfaceOrientation interfaceOrientation) {
    FrameOrientation frameOrientation = FrameOrientationPortrait; // provide a default to keep static analyzer happy
    switch(interfaceOrientation) {
        case UIInterfaceOrientationPortrait:
            frameOrientation = FrameOrientationPortrait;
            break;
        case UIInterfaceOrientationPortraitUpsideDown:
            frameOrientation = FrameOrientationPortraitUpsideDown;
            break;
        case UIInterfaceOrientationLandscapeLeft:
            frameOrientation = FrameOrientationLandscapeLeft;
            break;
        case UIInterfaceOrientationLandscapeRight:
            frameOrientation = FrameOrientationLandscapeRight;
            break;
        default:
            frameOrientation = FrameOrientationPortrait;
            break;
    }
    return frameOrientation;
}

static inline CGRect CGRectWithCVMRect(cvm_rect rect) {
    return CGRectMake(rect.x, rect.y, rect.w, rect.h);
}

static inline CGRect CGRectWithRotatedCVMRect(cvm_rect rect) {
    return CGRectMake(rect.y, rect.x, rect.h, rect.w);
}
