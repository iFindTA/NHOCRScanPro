//
//  cvm_constants.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#ifndef NHOpenCVPro_cvm_constants_h
#define NHOpenCVPro_cvm_constants_h

#define kRotationAnimationDuration 0.2f

#define kCreditCardTargetWidth 428
#define kCreditCardTargetHeight 270

#define kPortraitSampleWidth 480
#define kPortraitSampleHeight 640

#define kNumberWidth 19
#define kNumberHeight 27

#define kPortraitVerticalInset ((kPortraitSampleHeight - kCreditCardTargetHeight) / 2)
#define kPortraitVerticalPercentInset ((float)kPortraitVerticalInset / (float)kPortraitSampleHeight)
#define kPortraitHorizontalInset ((kPortraitSampleWidth - kCreditCardTargetWidth) / 2)
#define kPortraitHorizontalPercentInset ((float)kPortraitHorizontalInset / (float)kPortraitSampleWidth)

#define kLandscapeSampleWidth kPortraitSampleHeight
#define kLandscapeSampleHeight kPortraitSampleWidth

#define kLandscapeVerticalInset ((kLandscapeSampleHeight - kCreditCardTargetHeight) / 2)
#define kLandscapeVerticalPercentInset ((float)kLandscapeVerticalInset / (float)kLandscapeSampleHeight)
#define kLandscapeHorizontalInset ((kLandscapeSampleWidth - kCreditCardTargetWidth) / 2)
#define kLandscapeHorizontalPercentInset ((float)kLandscapeHorizontalInset / (float)kLandscapeSampleWidth)

#endif
