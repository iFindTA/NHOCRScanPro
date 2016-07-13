//
//  NHScanVCR.m
//  NHOCRScanPro
//
//  Created by hu jiaju on 15/7/30.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHScanVCR.h"
#import "NHCameraView.h"
#import "NHIplImage.h"

#pragma mark -- Scan View --

@interface NHScanVCR ()<NHVideoStreamDelegate>

@property (nonatomic, strong) NHCameraView *cameraView;
@property (nonatomic, strong) UIImageView *retImgView;
@property (nonatomic, strong) NSMutableArray *retImgs;

@end

@implementation NHScanVCR

-(void)viewDidLoad{
    [super viewDidLoad];
    
    self.title = @"VideoStream";
    self.view.backgroundColor = [UIColor whiteColor];
    
    self.cameraView = [[NHCameraView alloc] initWithFrame:self.view.bounds
                                                 delegate:self];
    [self.view addSubview:self.cameraView];
    [self.cameraView willAppear];
    
    CGRect infoRect = CGRectMake(0, 64, 320, 150);
    _retImgView = [[UIImageView alloc] initWithFrame:infoRect];
    _retImgView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_retImgView];
    
    int num = 20;int dis = 5;
    int width = (320-dis*(num-1))/num;
    int height = width*1.5;
    _retImgs = [[NSMutableArray alloc] initWithCapacity:0];
    @autoreleasepool {
        for (int i = 0;i < num; i++) {
            infoRect = CGRectMake((width+dis)*i, 64, width, height);
            UIImageView *img = [[UIImageView alloc] initWithFrame:infoRect];
            img.contentMode = UIViewContentModeScaleAspectFit;
            [_retImgs addObject:img];
            [self.view addSubview:img];
        }
    }
    
    [self performSelector:@selector(startSession) withObject:nil afterDelay:0.0f];
}

-(void)viewWillDisappear:(BOOL)animated{
    [super viewWillDisappear:animated];
    [self stopSession];
}

#pragma mark - Video session start/stop

- (void)startSession {
    if (self.cameraView) {
        NSLog(@"Starting CameraViewController session");
        
        [self.cameraView startVideoStreamSession];
    }
}

- (void)stopSession {
    if (self.cameraView) {
        NSLog(@"Stopping CameraViewController session");
        [self.cameraView stopVideoStreamSession];
    }
    
}

- (void)videoStream:(NHVideoStream *)stream didProcessFrame:(NHVideoFrame *)processedFrame{
    //NSLog(@"received video streams !");
    if ([processedFrame foundAllEdges]) {
        NSLog(@"almost found rect card !");
        //UIImage *dstImg = [processedFrame imageWithGrayscale:true];
        UIImage *dstImg = [processedFrame.retImg UIImage];
        dispatch_async(dispatch_get_main_queue(), ^{
            _retImgView.image = dstImg;
        });
    }
}

@end
