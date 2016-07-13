//
//  NHIplImage.m
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/13.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHIplImage.h"
#import "NHConstants.h"
#import "cvm.h"
#import "cvm_olm.h"
#import <opencv2/imgproc/imgproc_c.h>

using namespace opencvm;
void cvm_deinterleave_uint8_c2(IplImage *interleaved, IplImage **channel1, IplImage **channel2);

@interface NHIplImage ()

@property (nonatomic, assign, readwrite) IplImage *image;
@property (assign) Cvm cvm;

@end

@implementation NHIplImage

+ (NHIplImage *)imageWithSize:(CvSize)size depth:(int)depth channels:(int)channels {
    IplImage *newImage = cvCreateImage(size, depth, channels);
    return [self imageWithIplImage:newImage];
}

+ (NHIplImage *)imageFromYCbCrBuffer:(CVImageBufferRef)imageBuffer plane:(size_t)plane {
    char *planeBaseAddress = (char *)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, plane);
    
    size_t width = CVPixelBufferGetWidthOfPlane(imageBuffer, plane);
    size_t height = CVPixelBufferGetHeightOfPlane(imageBuffer, plane);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, plane);
    
    int numChannels = plane == Y_PLANE ? 1 : 2;
    IplImage *colocatedImage = cvCreateImageHeader(cvSize((int)width, (int)height), IPL_DEPTH_8U, numChannels);
    colocatedImage->imageData = planeBaseAddress;
    colocatedImage->widthStep = (int)bytesPerRow;
    
    return [self imageWithIplImage:colocatedImage];
}

+ (NHIplImage *)imageWithIplImage:(IplImage *)anImage {
    return [[self alloc] initWithIplImage:anImage];
}

- (id)initWithIplImage:(IplImage *)anImage {
    if((self = [super init])) {
        self.image = anImage;
    }
    return self;
}

+ (NHIplImage *)rgbImageWithY:(NHIplImage *)y cb:(NHIplImage *)cb cr:(NHIplImage *)cr {
    IplImage *rgb = NULL;
    cvm_YCbCr_to_RGB(y.image, cb.image, cr.image, &rgb);
    return [self imageWithIplImage:rgb];
}


- (NSArray *)split {
    if(self.image->nChannels == 1) {
        return [NSArray arrayWithObject:self];
    }
    assert(self.image->nChannels == 2); // not implemented for more
    IplImage *channel1;
    IplImage *channel2;
    _cvm.cvm_deinterleave_uint8_c2(self.image, &channel1, &channel2);
    NHIplImage *image1 = [[self class] imageWithIplImage:channel1];
    NHIplImage *image2 = [[self class] imageWithIplImage:channel2];
    return [NSArray arrayWithObjects:image1, image2, nil];
}

- (NSString *)description {
    CvSize s = self.cvSize;
    return [NSString stringWithFormat:@"<NHIplImage %p: %ix%i>", self, s.width, s.height];
}

- (void)dealloc {
    cvReleaseImage(&image);
}

- (IplImage *)image {
    return image;
}

- (CvSize)cvSize {
    return cvGetSize(self.image);
}

- (CGSize)cgSize {
    CvSize s = self.cvSize;
    return CGSizeMake(s.width, s.height);
}

- (CvRect)cvRect {
    CvSize s = self.cvSize;
    return cvRect(0, 0, s.width - 1, s.height - 1);
}

- (void)setImage:(IplImage *)newImage {
    assert(image == NULL);
    image = newImage;
}

- (IplImage *)iplImage{
    return self.image;
}

- (UIImage *)UIImage {
    CGColorSpaceRef colorSpace = NULL;
    if(self.image->nChannels == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else if(self.image->nChannels == 3) {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    //int depth = self.image->depth & ~IPL_DEPTH_SIGN;
    int depth = self.image->depth ;
    NSData *data = [NSData dataWithBytes:self.image->imageData length:self.image->imageSize];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(self.image->width,
                                        self.image->height,
                                        depth,
                                        depth * self.image->nChannels,
                                        self.image->widthStep,
                                        colorSpace,
                                        kCGImageAlphaNone | kCGBitmapByteOrderDefault,
                                        provider,
                                        NULL,
                                        false,
                                        kCGRenderingIntentDefault);
    UIImage *ret = [UIImage imageWithCGImage:imageRef scale:1.0 orientation:UIImageOrientationUp];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return ret;
}

- (NHIplImage *)copyCropped:(CvRect)roi {
    return [self copyCropped:roi destSize:cvGetSize(self.image)];
}

- (NHIplImage *)copyCropped:(CvRect)roi destSize:(CvSize)destSize {
    CvRect currentROI = cvGetImageROI(self.image);
    cvSetImageROI(self.image, roi);
    IplImage *copied = cvCreateImage(destSize, self.image->depth, self.image->nChannels);
    
    if (roi.width == destSize.width && roi.height == destSize.height) {
        cvCopy(self.image, copied, NULL);
    }
    else {
        cvResize(self.image, copied, CV_INTER_LINEAR);
    }
    
    cvSetImageROI(self.image, currentROI);
    return [NHIplImage imageWithIplImage:copied];
}

@end
