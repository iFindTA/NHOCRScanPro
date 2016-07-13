//
//  NHVideoStream.m
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHVideoStream.h"
#import "NHVideoFrame.h"
#import "cvm_olm.h"

#define kCaptureSessionDefaultPresetResolution AVCaptureSessionPreset640x480
#define kVideoQueueName "io.card.ios.videostream"

#define kMinTimeIntervalForAutoFocusOnce 2

#define kIsoThatSuggestsMoreTorchlightThanWeReallyNeed 250
#define kRidiculouslyHighIsoSpeed 10000
#define kMinimalTorchLevel 0.05f
#define kCoupleOfHours 10000

#define kMinNumberOfFramesScannedToDeclareUnscannable 100

@interface NHVideoStream ()<AVCaptureVideoDataOutputSampleBufferDelegate>{
@private
    cvm_context *cvm;
}

@property(nonatomic, assign, readwrite) BOOL running;
@property(nonatomic, assign, readwrite) BOOL wasRunningBeforeBeingBackgrounded;
@property(nonatomic, assign, readwrite) BOOL didEndGeneratingDeviceOrientationNotifications;
@property(assign, readwrite) UIInterfaceOrientation interfaceOrientation; // intentionally atomic -- video frames are processed on a different thread
@property(nonatomic, strong, readwrite) AVCaptureVideoPreviewLayer *previewLayer;
@property(nonatomic, strong, readwrite) AVCaptureSession *captureSession;
@property(nonatomic, strong, readwrite) AVCaptureDevice *camera;
@property(nonatomic, strong, readwrite) AVCaptureDeviceInput *cameraInput;
@property(nonatomic, strong, readwrite) AVCaptureVideoDataOutput *videoOutput;

@property (nonatomic, assign, readwrite) NSTimeInterval lastAutoFocusOnceTime;
@property (nonatomic, assign, readwrite) BOOL           currentlyAdjustingFocus;
@property (nonatomic, assign, readwrite) BOOL           currentlyAdjustingExposure;
@property (nonatomic, assign, readwrite) NSTimeInterval lastChangeSignal;
@property (nonatomic, assign, readwrite) BOOL           lastChangeTorchStateToOFF;

// This semaphore is intended to prevent a crash which was recorded with this exception message:
// "AVCaptureSession can't stopRunning between calls to beginConfiguration / commitConfiguration"
@property(nonatomic, strong, readwrite) dispatch_semaphore_t cameraConfigurationSemaphore;

@end

@implementation NHVideoStream

- (void)dealloc {
    [self stopSession]; // just to be safe
    
    if (!self.didEndGeneratingDeviceOrientationNotifications) {
        [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
    }
    
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    //cvm_context_destroy(cvm), cvm = NULL;
    
#if !__has_feature(objc_arc)
    [super dealloc];
#endif
}

- (id)init {
    if((self = [super init])) {
        _interfaceOrientation = (UIInterfaceOrientation)UIDeviceOrientationUnknown;
        //_scanner = [[CardIOCardScanner alloc] init];
        _cameraConfigurationSemaphore = dispatch_semaphore_create(1); // parameter of `1` implies "allow access to only one thread at a time"

        _captureSession = [[AVCaptureSession alloc] init];
        _camera = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        _previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
        //cvm = cvm_context_create();
        //初始化神经网络
        //self.m_scanner =  CvScan::CvScan();
        NSString *ann_ns=[[NSBundle mainBundle]pathForResource:@"OCR" ofType:@"xml"];
        //NSString *svm_ns=[[NSBundle mainBundle]pathForResource:@"svm" ofType:@"xml"];
        std::string annpath=[ann_ns UTF8String];
        self.m_scanner.setModelPath(annpath);
        self.m_scanner.LoadANNModel(annpath);
    }
    
    return self;
}

#pragma mark - Orientation

- (void)willAppear {
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(didReceiveBackgroundingNotification:)
                                                 name:UIApplicationWillResignActiveNotification
                                               object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(didReceiveForegroundingNotification:)
                                                 name:UIApplicationDidBecomeActiveNotification
                                               object:nil];
    
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(didReceiveDeviceOrientationNotification:)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:[UIDevice currentDevice]];
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
    [self didReceiveDeviceOrientationNotification:nil];
    
    // If we ever want to use higher resolution images, this is a good place to do that.
    //    if ([self.captureSession canSetSessionPreset:AVCaptureSessionPreset1920x1080]) {
    //      self.captureSession.sessionPreset = AVCaptureSessionPreset1920x1080;
    //    }
    //    else
    //    if ([self.captureSession canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
    //      self.captureSession.sessionPreset = AVCaptureSessionPreset1280x720;
    //    }
    //  }
}

- (void)willDisappear {
    self.didEndGeneratingDeviceOrientationNotifications = true;
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)didReceiveDeviceOrientationNotification:(NSNotification *)notification {
    UIInterfaceOrientation newInterfaceOrientation;
    switch([UIDevice currentDevice].orientation) {
        case UIDeviceOrientationPortrait:
            newInterfaceOrientation = UIInterfaceOrientationPortrait;
            break;
        case UIDeviceOrientationPortraitUpsideDown:
            newInterfaceOrientation = UIInterfaceOrientationPortraitUpsideDown;
            break;
        case UIDeviceOrientationLandscapeLeft:
            newInterfaceOrientation = UIInterfaceOrientationLandscapeRight;
            break;
        case UIDeviceOrientationLandscapeRight:
            newInterfaceOrientation = UIInterfaceOrientationLandscapeLeft;
            break;
        default:
            newInterfaceOrientation = UIInterfaceOrientationPortrait;
            break;
    }
    
    if ([self.delegate respondsToSelector:@selector(isSupportedOverlayOrientation:)] &&
        [self.delegate respondsToSelector:@selector(defaultSupportedOverlayOrientation)]) {
        if (![self.delegate isSupportedOverlayOrientation:newInterfaceOrientation]) {
            if ([self.delegate isSupportedOverlayOrientation:self.interfaceOrientation]) {
                newInterfaceOrientation = self.interfaceOrientation;
            }
            else {
                UIInterfaceOrientation orientation = [self.delegate defaultSupportedOverlayOrientation];
                if (orientation != (UIInterfaceOrientation)UIDeviceOrientationUnknown) {
                    newInterfaceOrientation = orientation;
                }
            }
        }
    }
    
    if (newInterfaceOrientation != self.interfaceOrientation) {
        self.interfaceOrientation = newInterfaceOrientation;
    }
}

#pragma mark - Camera configuration changing

- (BOOL)changeCameraConfiguration:(void(^)())changeBlock {
    dispatch_semaphore_wait(self.cameraConfigurationSemaphore, DISPATCH_TIME_FOREVER);
    
    BOOL success = NO;
    NSError *lockError = nil;
    [self.captureSession beginConfiguration];
    [self.camera lockForConfiguration:&lockError];
    if(!lockError) {
        changeBlock();
        [self.camera unlockForConfiguration];
        success = YES;
    }
    
    [self.captureSession commitConfiguration];
    
    dispatch_semaphore_signal(self.cameraConfigurationSemaphore);
    
    return success;
}

#pragma mark - Torch

- (BOOL)hasTorch {
    return [self.camera hasTorch] &&
    [self.camera isTorchModeSupported:AVCaptureTorchModeOn] &&
    [self.camera isTorchModeSupported:AVCaptureTorchModeOff] &&
    self.camera.torchAvailable;
}

- (BOOL)canSetTorchLevel {
    return [self.camera hasTorch] && [self.camera respondsToSelector:@selector(setTorchModeOnWithLevel:error:)];
}

- (BOOL)torchIsOn {
    return self.camera.torchMode == AVCaptureTorchModeOn;
}

- (BOOL)setTorchOn:(BOOL)torchShouldBeOn {
    return [self changeCameraConfiguration:^{
        AVCaptureTorchMode newTorchMode = torchShouldBeOn ? AVCaptureTorchModeOn : AVCaptureTorchModeOff;
        [self.camera setTorchMode:newTorchMode];
    }];
}

- (BOOL)setTorchModeOnWithLevel:(float)torchLevel {
    __block BOOL torchSuccess = NO;
    BOOL success = [self changeCameraConfiguration:^{
        NSError *error;
        torchSuccess = [self.camera setTorchModeOnWithLevel:torchLevel error:&error];
    }];
    
    return success && torchSuccess;
}

#pragma mark - Focus

- (BOOL)hasAutofocus {
    return [self.camera isFocusModeSupported:AVCaptureFocusModeAutoFocus];
}

- (void)refocus {
    [self autofocusOnce];
    [self performSelector:@selector(resumeContinuousAutofocusing) withObject:nil afterDelay:0.1f];
}

- (void)autofocusOnce {
    [self changeCameraConfiguration:^{
        if([self.camera isFocusModeSupported:AVCaptureFocusModeAutoFocus]) {
            [self.camera setFocusMode:AVCaptureFocusModeAutoFocus];
        }
    }];
}

- (void)resumeContinuousAutofocusing {
    [self changeCameraConfiguration:^{
        if([self.camera isFocusModeSupported:AVCaptureFocusModeContinuousAutoFocus]) {
            [self.camera setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
        }
    }];
}

#pragma mark - Session

// Consistent with <https://devforums.apple.com/message/887783#887783>, under iOS 7 it
// appears that our captureSession's input and output linger in memory even after the
// captureSession itself is dealloc'ed, unless we explicitly call removeInput: and
// removeOutput:.
//
// Moreover, it can be a long time from when we are fully released until we are finally dealloc'ed.
//
// The result is that if a user triggers a series of camera sessions, especially without long pauses
// in between, we start clogging up memory with our cameraInput and videoOutput objects.
//
// So I've now moved the creation and adding of input and output objects from [self init] to
// [self startSession]. And in [self stopSession] I'm now removing those objects.
// This seems to have solved the problem (for now, anyways).

- (BOOL)addInputAndOutput {
    NSError *sessionError = nil;
    _cameraInput = [AVCaptureDeviceInput deviceInputWithDevice:self.camera error:&sessionError];
    if(sessionError || !self.cameraInput) {
        NSLog(@"CardIO camera input error: %@", sessionError);
        return NO;
    }
    
    [self.captureSession addInput:self.cameraInput];
    self.captureSession.sessionPreset = kCaptureSessionDefaultPresetResolution;
    
    _videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    NSDictionary *videoOutputSettings = [NSDictionary dictionaryWithObject:[NSNumber numberWithUnsignedInteger:kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] forKey:(NSString *)kCVPixelBufferPixelFormatTypeKey];
    [self.videoOutput setVideoSettings:videoOutputSettings];
    self.videoOutput.alwaysDiscardsLateVideoFrames = YES;
    // NB: DO NOT USE minFrameDuration. minFrameDuration causes focusing to
    // slow down dramatically, which causes significant ux pain.
    dispatch_queue_t queue = dispatch_queue_create(kVideoQueueName, NULL);
    [self.videoOutput setSampleBufferDelegate:self queue:queue];
    
    [self.captureSession addOutput:self.videoOutput];
    
    return YES;
}

- (void)removeInputAndOutput {
    [self.captureSession removeInput:self.cameraInput];
    [self.videoOutput setSampleBufferDelegate:nil queue:NULL];
    [self.captureSession removeOutput:self.videoOutput];
}

- (void)startSession {
    if ([self addInputAndOutput]) {
        [self.camera addObserver:self forKeyPath:@"adjustingFocus" options:(NSKeyValueObservingOptionNew | NSKeyValueObservingOptionInitial) context:nil];
        [self.camera addObserver:self forKeyPath:@"adjustingExposure" options:(NSKeyValueObservingOptionNew | NSKeyValueObservingOptionInitial) context:nil];
        [self.captureSession startRunning];
        
        [self changeCameraConfiguration:^{
            if ([self.camera respondsToSelector:@selector(isAutoFocusRangeRestrictionSupported)]) {
                if(self.camera.autoFocusRangeRestrictionSupported) {
                    self.camera.autoFocusRangeRestriction = AVCaptureAutoFocusRangeRestrictionNear;
                }
            }
            if ([self.camera respondsToSelector:@selector(isFocusPointOfInterestSupported)]) {
                if(self.camera.focusPointOfInterestSupported) {
                    self.camera.focusPointOfInterest = CGPointMake(0.5, 0.5);
                }
            }
        }];
        self.running = YES;
    }
}

- (void)stopSession {
    if (self.running) {

        [self changeCameraConfiguration:^{
            // restore default focus range
            if ([self.camera respondsToSelector:@selector(isAutoFocusRangeRestrictionSupported)]) {
                if(self.camera.autoFocusRangeRestrictionSupported) {
                    self.camera.autoFocusRangeRestriction = AVCaptureAutoFocusRangeRestrictionNone;
                }
            }
        }];
        
        dispatch_semaphore_wait(self.cameraConfigurationSemaphore, DISPATCH_TIME_FOREVER);
        
        [self.camera removeObserver:self forKeyPath:@"adjustingExposure"];
        [self.camera removeObserver:self forKeyPath:@"adjustingFocus"];
        [self.captureSession stopRunning];
        [self removeInputAndOutput];
        
        self.running = NO;
        
        dispatch_semaphore_signal(self.cameraConfigurationSemaphore);
    }
}

- (void)sendFrameToDelegate:(NHVideoFrame *)frame {
    // Due to threading, we can receive frames after we've stopped running.
    // Clean this up for our delegate.
    if(self.running) {
        [self.delegate videoStream:self didProcessFrame:frame];
    }
    else {
        NSLog(@"STRAY FRAME!!! wasted processing. we are sad.");
    }
}

#pragma mark - Key-Value Observing

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context {
    if ([keyPath isEqualToString:@"adjustingFocus"]) {
        self.currentlyAdjustingFocus = [change[NSKeyValueChangeNewKey] boolValue];
    }
    else if ([keyPath isEqualToString:@"adjustingExposure"]) {
        self.currentlyAdjustingExposure = [change[NSKeyValueChangeNewKey] boolValue];
    }
}

#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate methods

#ifdef __IPHONE_6_0 // Compile-time check for the time being, so our code still compiles with the fully released toolchain
- (void)captureOutput:(AVCaptureOutput *)captureOutput didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
}
#endif

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    @autoreleasepool {
        NHVideoFrame *frame = [[NHVideoFrame alloc] initWithSampleBuffer:sampleBuffer interfaceOrientation:self.interfaceOrientation];
        frame.cvm = cvm;
        //frame.m_scanner = self.m_scanner;
        //frame.scanExpiry = self.config.scanExpiry;
        //frame.detectionMode = self.config.detectionMode;
        
        if (self.running) {
            if (!self.currentlyAdjustingFocus ) {
                if ([self canSetTorchLevel]) {
                    frame.calculateBrightness = YES;
                    frame.torchIsOn = [self torchIsOn];
                }
                
                NSDictionary *exifDict = (__bridge NSDictionary *)((CFDictionaryRef)CMGetAttachment(sampleBuffer, (CFStringRef)@"{Exif}", NULL));
                if (exifDict != nil) {
                    frame.isoSpeed = [exifDict[@"ISOSpeedRatings"][0] integerValue];
                    frame.shutterSpeed = [exifDict[@"ShutterSpeedValue"] floatValue];
                }
                else {
                    frame.isoSpeed = kRidiculouslyHighIsoSpeed;
                    frame.shutterSpeed = 0;
                }
                
                [frame process];
            }
            
            [self performSelectorOnMainThread:@selector(sendFrameToDelegate:) withObject:frame waitUntilDone:NO];
            
            // Autofocus
            BOOL didAutoFocus = NO;
            if (!self.currentlyAdjustingFocus && frame.focusSucks && [self hasAutofocus]) {
                NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
                if (now - self.lastAutoFocusOnceTime > kMinTimeIntervalForAutoFocusOnce) {
                    self.lastAutoFocusOnceTime = now;
                    NSLog(@"Auto-triggered focusing");
                    [self autofocusOnce];
                    [self performSelector:@selector(resumeContinuousAutofocusing) withObject:nil afterDelay:0.1f];
                    didAutoFocus = YES;
                }
            }
            
            // Auto-torch
            if (!self.currentlyAdjustingFocus && !didAutoFocus && !self.currentlyAdjustingExposure && [self canSetTorchLevel]) {
                NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
                BOOL changeTorchState = NO;
                BOOL changeTorchStateToOFF = NO;
                if (frame.brightnessHigh) {
                    if ([self torchIsOn]) {
                        changeTorchState = YES;
                        changeTorchStateToOFF = YES;
                    }
                }
                else {
                    if (frame.brightnessLow) {
                        if (![self torchIsOn] && frame.isoSpeed > kIsoThatSuggestsMoreTorchlightThanWeReallyNeed) {
                            changeTorchState = YES;
                            changeTorchStateToOFF = NO;
                        }
                    }
                    else if ([self torchIsOn]) {
                        if (frame.isoSpeed < kIsoThatSuggestsMoreTorchlightThanWeReallyNeed) {
                            changeTorchState = YES;
                            changeTorchStateToOFF = YES;
                        }
                    }
                }
                
                // Require at least two consecutive change signals in the same direction, over at least one second.
                
                // Note: if self.lastChangeSignal == 0.0, then we've just entered camera view.
                // In that case, lastChangeTorchStateToOFF == NO, and so turning ON the torch won't wait that second.
                
                if (changeTorchState) {
                    if (changeTorchStateToOFF == self.lastChangeTorchStateToOFF) {
                        if (now - self.lastChangeSignal > 1) {
                            NSLog(@"Automatic torch change");
                            if (changeTorchStateToOFF) {
                                [self setTorchOn:NO];
                            }
                            else {
                                [self setTorchModeOnWithLevel:kMinimalTorchLevel];
                            }
                            self.lastChangeSignal = now + kCoupleOfHours;
                        }
                        else {
                            self.lastChangeSignal = MIN(self.lastChangeSignal, now);
                        }
                    }
                    else {
                        self.lastChangeSignal = now;
                        self.lastChangeTorchStateToOFF = changeTorchStateToOFF;
                    }
                }
                else {
                    self.lastChangeSignal = now + kCoupleOfHours;
                }
            }
        }
    }
}

#pragma mark - Suspend/Resume when app is backgrounded/foregrounded

- (void)didReceiveBackgroundingNotification:(NSNotification *)notification {
    self.wasRunningBeforeBeingBackgrounded = self.running;
    [self stopSession];
    ///to stop GPU Render
    //cvm_prepare_for_backgrounding(cvm);
}

- (void)didReceiveForegroundingNotification:(NSNotification *)notification {
    if (self.wasRunningBeforeBeingBackgrounded) {
        [self startSession];
    }
}

@end
