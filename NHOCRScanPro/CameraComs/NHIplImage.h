//
//  NHIplImage.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/13.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>
#import <opencv2/core/types_c.h>

#define Y_PLANE 0
#define CBCR_PLANE 1

@interface NHIplImage : NSObject{
@private
    IplImage *image;
}

+ (NHIplImage *)imageWithSize:(CvSize)size depth:(int)depth channels:(int)channels;
+ (NHIplImage *)imageWithIplImage:(IplImage *)anImage;
- (id)initWithIplImage:(IplImage *)anImage;

+ (NHIplImage *)imageFromYCbCrBuffer:(CVImageBufferRef)imageBuffer plane:(size_t)plane;

- (NHIplImage *)copyCropped:(CvRect)roi;
- (NHIplImage *)copyCropped:(CvRect)roi destSize:(CvSize)destSize;

- (NSArray *)split;

+ (NHIplImage *)rgbImageWithY:(NHIplImage *)y cb:(NHIplImage *)cb cr:(NHIplImage *)cr;

- (UIImage *)UIImage;
- (IplImage *)iplImage;

@property(nonatomic, assign, readonly) IplImage *image;
@property(nonatomic, assign, readonly) CvSize cvSize;
@property(nonatomic, assign, readonly) CGSize cgSize;
@property(nonatomic, assign, readonly) CvRect cvRect;

@end
