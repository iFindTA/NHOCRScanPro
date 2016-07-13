//
//  NHShutterView.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface NHShutterView : UIView

- (void)setOpen:(BOOL)shouldBeOpen animated:(BOOL)animated duration:(CFTimeInterval)duration;

@property(nonatomic, assign, readwrite) BOOL open;

@end
