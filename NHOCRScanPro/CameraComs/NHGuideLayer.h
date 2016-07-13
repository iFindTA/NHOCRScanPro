//
//  NHGuideLayer.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>

@protocol NHGuideLayerDelegate <NSObject>
- (void)guideLayerDidLayout:(CGRect)internalGuideFrame;
@end

@class NHVideoFrame;

@interface NHGuideLayer : CALayer

- (id)initWithDelegate:(id<NHGuideLayerDelegate>)guideLayerDelegate;

- (CGRect)guideFrame;

- (void)showCardFound:(BOOL)found;

- (void)didRotateToDeviceOrientation:(UIDeviceOrientation)deviceOrientation;

@property(nonatomic, strong, readwrite) UIColor *guideColor;
@property(nonatomic, strong, readwrite) NHVideoFrame *videoFrame;
@property(nonatomic, assign, readwrite) CFTimeInterval animationDuration;
@property(nonatomic, assign, readwrite) UIDeviceOrientation deviceOrientation;
@property(nonatomic, strong, readwrite) CAGradientLayer *fauxCardLayer;

@end
