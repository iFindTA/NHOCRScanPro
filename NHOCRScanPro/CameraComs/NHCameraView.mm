//
//  NHCameraView.m
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHCameraView.h"
#import "NHOrientation.h"
#import "NHConstants.h"
#import "NHShutterView.h"
#import "cvm_constants.h"
#import "cvm_scan.h"
#import "NHIplImage.h"
#import "NHGPUTransformFilter.h"

#define kLogoAlpha 0.6f
#define kGuideLayerTextAlpha 0.6f

#define kGuideLayerTextColor [UIColor colorWithWhite:1.0f alpha:kGuideLayerTextAlpha]
#define kLabelVisibilityAnimationDuration 0.3f
#define kRotationLabelShowDelay (kRotationAnimationDuration + 0.1f)

#define kStandardInstructionsFontSize 18.0f
#define kMinimumInstructionsFontSize (kStandardInstructionsFontSize / 2)

using namespace opencv_scan;
void LoadANNModel(std::string s);
void LoadSVMModel(std::string s);
void charsSegement(IplImage *src, std::vector<cv::Mat> &vector);
void charsImgSegement(IplImage *src, vector<IplImage*> &vector);
string charsIdentify(Mat input);

@interface NHCameraView ()

@property(nonatomic, strong, readonly) NHGuideLayer *cardGuide;
@property(nonatomic, strong, readwrite) UILabel *guideLayerLabel;
@property(nonatomic, strong, readwrite) NHShutterView *shutter;
@property(nonatomic, strong, readwrite) NHVideoStream *videoStream;
@property(nonatomic, strong, readwrite) UIButton *lightButton;
@property(nonatomic, strong, readwrite) UIImageView *logoView;
@property(nonatomic, assign, readwrite) UIDeviceOrientation deviceOrientation;
@property(nonatomic, assign, readwrite) BOOL rotatingInterface;
@property(nonatomic, assign, readwrite) BOOL videoStreamSessionWasRunningBeforeRotation;
//@property(nonatomic, strong, readwrite) CardIOConfig *config;
@property(nonatomic, assign, readwrite) BOOL hasLaidoutCameraButtons;
@property (nonatomic, strong) NHGPUTransformFilter *gpuFilter;

@end

@implementation NHCameraView

+ (CGRect)previewRectWithinSize:(CGSize)size landscape:(BOOL)landscape {
    CGSize contents;
    if(landscape) {
        contents = CGSizeMake(kLandscapeSampleWidth, kLandscapeSampleHeight);
    } else {
        contents = CGSizeMake(kPortraitSampleWidth, kPortraitSampleHeight);
    }
    CGRect contentsRect = aspectFit(contents, size);
    return CGRectFlooredToNearestPixel(contentsRect);
}

- (id)initWithFrame:(CGRect)frame {
    [NSException raise:@"Wrong initializer" format:@"CardIOCameraView's designated initializer is initWithFrame:delegate:config:"];
    return nil;
}

- (id)initWithFrame:(CGRect)frame delegate:(id<NHVideoStreamDelegate>)delegate {
    self = [super initWithFrame:frame];
    if(self) {
        _deviceOrientation = UIDeviceOrientationUnknown;
        
        self.autoresizingMask = UIViewAutoresizingNone;
        self.backgroundColor = [UIColor clearColor];
        self.clipsToBounds = YES;
        
        _delegate = delegate;
        //_config = config;
        
        _videoStream = [[NHVideoStream alloc] init];
        //self.videoStream.config = config;
        self.videoStream.delegate = self;
        self.videoStream.previewLayer.needsDisplayOnBoundsChange = YES;
        self.videoStream.previewLayer.contentsGravity = kCAGravityResizeAspect;
        self.videoStream.previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
        
        // These settings are helpful when debugging rotation/bounds/rendering issues:
        // self.videoStream.previewLayer.backgroundColor = [UIColor yellowColor].CGColor;
        
        // Preview of the camera image
        [self.layer addSublayer:self.videoStream.previewLayer];
        
        // Guide layer shows card guide edges and other progress feedback directly related to the camera contents
        _cardGuide = [[NHGuideLayer alloc] initWithDelegate:self];
        self.cardGuide.contentsGravity = kCAGravityResizeAspect;
        self.cardGuide.needsDisplayOnBoundsChange = YES;
        self.cardGuide.animationDuration = kRotationAnimationDuration;
        self.cardGuide.deviceOrientation = self.deviceOrientation;
        //self.cardGuide.guideColor = config.guideColor;
        [self.layer addSublayer:self.cardGuide];
        
        _guideLayerLabel = [[UILabel alloc] initWithFrame:CGRectZero];
        self.guideLayerLabel.text = @"请将卡片置于检测框内！";
        self.guideLayerLabel.textAlignment = NSTextAlignmentCenter;
        self.guideLayerLabel.backgroundColor = [UIColor clearColor];
        self.guideLayerLabel.textColor = kGuideLayerTextColor;
        self.guideLayerLabel.font = [UIFont fontWithName:@"Helvetica-Bold" size:kStandardInstructionsFontSize];
        self.guideLayerLabel.numberOfLines = 0;
        [self addSubview:self.guideLayerLabel];
        
        // Shutter view for shutter-open animation
        _shutter = [[NHShutterView alloc] initWithFrame:CGRectZero];
        [self.shutter setOpen:NO animated:NO duration:0];
        [self addSubview:self.shutter];
        
        // Tap-to-refocus support
        if([self.videoStream hasAutofocus]) {
            UITapGestureRecognizer *touch = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(refocus)];
            [self addGestureRecognizer:touch];
        }
        
        _gpuFilter = [[NHGPUTransformFilter alloc] initWithSize:CGSizeMake(12, 18)];
    }
    return self;
}

- (void)startVideoStreamSession {
    [self.videoStream startSession];
    
    // If we don't do this, then when the torch was on, and the card read failed,
    // it still shows as on when this view is re-displayed, even though the ending
    // of the session turned it off.
    //[self updateLightButtonState];
}

- (void)stopVideoStreamSession {
    [self.videoStream stopSession];
    [self.shutter setOpen:NO animated:NO duration:0.0f];
    if (_gpuFilter) {
        [_gpuFilter finish];
    }
}

- (void)refocus {
    [self.videoStream refocus];
}

- (void)setSuppressFauxCardLayer:(BOOL)suppressFauxCardLayer {
    if(suppressFauxCardLayer) {
        self.cardGuide.fauxCardLayer.hidden = YES;
    }
    _suppressFauxCardLayer = suppressFauxCardLayer;
}


- (CGRect)guideFrame {
    return [self.cardGuide guideFrame];
}

- (CGRect)cameraPreviewFrame {
    CGRect cameraPreviewFrame = [[self class] previewRectWithinSize:self.bounds.size
                                                          landscape:UIInterfaceOrientationIsLandscape([UIApplication sharedApplication].statusBarOrientation)];
    return cameraPreviewFrame;
}

- (void)updateCameraOrientation {
    // We want the camera to appear natural. That means that the camera preview should match reality, orientationwise.
    // If the interfaceOrientation isn't portrait, then we need to rotate the preview precisely OPPOSITE the interface.
    CGFloat rotation = -orientationToRotation([UIApplication sharedApplication].statusBarOrientation);
    CATransform3D transform = CATransform3DIdentity;
    transform = CATransform3DRotate(transform, rotation, 0, 0, 1);
    //  NSLog(@"Updating camera orientation for interface orientation %@, device orientation %@: Rotation %f",
    //        INTERFACE_LANDSCAPE_OR_PORTRAIT([UIApplication sharedApplication].statusBarOrientation),
    //        DEVICE_LANDSCAPE_OR_PORTRAIT(self.deviceOrientation),
    //        rotation * 180 / M_PI);
    
    SuppressCAAnimate(^{
        self.videoStream.previewLayer.transform = transform;
    });
}

- (void)layoutSubviews {
    [self updateCameraOrientation];
    
    CGRect cameraPreviewFrame = [self cameraPreviewFrame];
    
    SuppressCAAnimate(^{
        if (!CGRectEqualToRect(self.videoStream.previewLayer.frame, cameraPreviewFrame)) {
            self.videoStream.previewLayer.frame = cameraPreviewFrame;
        }
        self.shutter.frame = cameraPreviewFrame;
        
        //[self layoutCameraButtons];
        
        self.cardGuide.frame = cameraPreviewFrame;
    });
}

- (void)didReceiveDeviceOrientationNotification:(NSNotification *)notification {
    UIDeviceOrientation newDeviceOrientation;
    switch ([UIDevice currentDevice].orientation) {
        case UIDeviceOrientationPortrait:
            newDeviceOrientation = UIDeviceOrientationPortrait;
            break;
        case UIDeviceOrientationPortraitUpsideDown:
            newDeviceOrientation = UIDeviceOrientationPortraitUpsideDown;
            break;
        case UIDeviceOrientationLandscapeLeft:
            newDeviceOrientation = UIDeviceOrientationLandscapeLeft;
            break;
        case UIDeviceOrientationLandscapeRight:
            newDeviceOrientation = UIDeviceOrientationLandscapeRight;
            break;
        default:
            if (self.deviceOrientation == UIDeviceOrientationUnknown) {
                newDeviceOrientation = UIDeviceOrientationPortrait;
            }
            else {
                newDeviceOrientation = self.deviceOrientation;
            }
            break;
    }
    
    if (![self isSupportedOverlayOrientation:(UIInterfaceOrientation)newDeviceOrientation]) {
        if ([self isSupportedOverlayOrientation:(UIInterfaceOrientation)self.deviceOrientation]) {
            newDeviceOrientation = self.deviceOrientation;
        }
        else {
            UIInterfaceOrientation orientation = [self defaultSupportedOverlayOrientation];
            if (orientation != (UIInterfaceOrientation)UIDeviceOrientationUnknown) {
                newDeviceOrientation = (UIDeviceOrientation)orientation;
            }
        }
    }
    
    if(newDeviceOrientation != self.deviceOrientation) {
        self.deviceOrientation = newDeviceOrientation;
        
        [self.cardGuide didRotateToDeviceOrientation:self.deviceOrientation];
        self.guideLayerLabel.hidden = YES;
        [self performSelector:@selector(showGuideLabel) withObject:nil afterDelay:kRotationLabelShowDelay];
        
        [self setNeedsLayout];
    }
}

- (void)showGuideLabel {
    // If we are rotating, let it stay hidden; the interface rotation cleanup code will re-show it
    if(!self.rotatingInterface) {
        self.guideLayerLabel.hidden = NO;
    }
}

- (void)orientGuideLayerLabel {
    InterfaceToDeviceOrientationDelta delta = orientationDelta([UIApplication sharedApplication].statusBarOrientation, self.deviceOrientation);
    CGFloat rotation = -rotationForOrientationDelta(delta); // undo the orientation delta
    self.guideLayerLabel.transform = CGAffineTransformMakeRotation(rotation);
}

#pragma mark - Orientation

- (void)setHidden:(BOOL)hidden {
    if (hidden != self.hidden) {
        if (hidden) {
            [self implicitStop];
            [super setHidden:hidden];
        }
        else {
            [super setHidden:hidden];
            [self implicitStart];
        }
    }
}

- (void)willMoveToSuperview:(UIView *)newSuperview {
    if (!newSuperview) {
        [self implicitStop];
    }
    [super willMoveToSuperview:newSuperview];
}

- (void)didMoveToSuperview {
    [super didMoveToSuperview];
    if (self.superview) {
        [self implicitStart];
    }
}

- (void)willMoveToWindow:(UIWindow *)newWindow {
    if (!newWindow) {
        [self implicitStop];
    }
    [super willMoveToWindow:newWindow];
}

- (void)didMoveToWindow {
    [super didMoveToWindow];
    if (self.window) {
        [self implicitStart];
    }
}

- (void)implicitStart {
    
    [self willAppear];
}

- (void)implicitStop {
    [self willDisappear];
}

- (void)willAppear {
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(didReceiveDeviceOrientationNotification:)
                                                 name:UIDeviceOrientationDidChangeNotification
                                               object:[UIDevice currentDevice]];
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
    
    self.deviceOrientation = UIDeviceOrientationUnknown;
    [self didReceiveDeviceOrientationNotification:nil];
    
    [self.videoStream willAppear];
    [self becomeFirstResponder];
}

- (void)willDisappear {
    [self.videoStream willDisappear];
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)vibrate {
    AudioServicesPlayAlertSound(kSystemSoundID_Vibrate);
}

#pragma mark - CardIOGuideLayerDelegate method

- (void)guideLayerDidLayout:(CGRect)internalGuideFrame {
    CGFloat width = MAX(internalGuideFrame.size.width, internalGuideFrame.size.height);
    CGFloat height = MIN(internalGuideFrame.size.width, internalGuideFrame.size.height);
    
    CGRect internalGuideRect = CGRectZeroWithSize(CGSizeMake(width, height));
    
    self.guideLayerLabel.bounds = internalGuideRect;
    [self.guideLayerLabel sizeToFit];
    
    CGRect cameraPreviewFrame = [self cameraPreviewFrame];
    self.guideLayerLabel.center = CGPointMake(CGRectGetMidX(cameraPreviewFrame), CGRectGetMidY(cameraPreviewFrame));
    
    internalGuideRect.size.height = 9999.9f;
    CGRect textRect = [self.guideLayerLabel textRectForBounds:internalGuideRect limitedToNumberOfLines:0];
    while (textRect.size.height > height && self.guideLayerLabel.font.pointSize > kMinimumInstructionsFontSize) {
        self.guideLayerLabel.font = [UIFont fontWithName:self.guideLayerLabel.font.fontName size:self.guideLayerLabel.font.pointSize - 1];
        textRect = [self.guideLayerLabel textRectForBounds:internalGuideRect limitedToNumberOfLines:0];
    }
    
    [self orientGuideLayerLabel];
}

#pragma mark - CardIOVideoStreamDelegate methods

- (void)videoStream:(NHVideoStream *)stream didProcessFrame:(NHVideoFrame *)processedFrame {
    [self.shutter setOpen:YES animated:YES duration:0.5f];
    
    // Hide instructions once we start to find edges
//    if (processedFrame.numEdgesFound < 0.05f) {
//        [UIView animateWithDuration:kLabelVisibilityAnimationDuration animations:^{self.guideLayerLabel.alpha = 1.0f;}];
//    } else if (processedFrame.numEdgesFound > 2.1f) {
//        [UIView animateWithDuration:kLabelVisibilityAnimationDuration animations:^{self.guideLayerLabel.alpha = 0.0f;}];
//    }
    if ([processedFrame foundAllEdges]) {
        [self vibrate];
        
        
        CvScan m_scanner = self.videoStream.m_scanner;
        if (!m_scanner.getBusyState()) {
            IplImage *matImg = processedFrame.retImg.iplImage;
            //UIImage *tmpImg = [processedFrame.retImg UIImage];
            
//            NSString *str = [self OCRImage:matImg];
//            NSLog(@"识别出的字符:%@",str);
//            return;
            
            NSString *path = [[NSBundle mainBundle] pathForResource:@"10_0.792674_gray_14967_5950_step5_recog_4_0_0.890217_0.705652" ofType:@"png"];
            vector<IplImage*>vector;
            
            m_scanner.charsImgSegement(matImg, vector);
            processedFrame.imgs = vector;
            int size = (int )vector.size();
            for (int i = 0; i < size;i++) {
                IplImage *src = vector[i];
                
                IplImage psrc = *new IplImage(*src);
                cv::Mat pmat = cv::cvarrToMat(&psrc);
                
                cv::Mat mat = imread([path UTF8String]);
                cv::cvtColor(mat, mat, CV_BGR2GRAY);
                
                string ret = m_scanner.charsIdentify(mat);
                cout<<"recognize ret : "<<ret<<endl;
                
                //NSString *str = [self OCRImage:src];
                //NSLog(@"识别出的字符:%@",str);
            }
        }
    }
    // Pass the video frame to the cardGuide so that it can update the edges
    self.cardGuide.videoFrame = processedFrame;
    
    
    [self.delegate videoStream:stream didProcessFrame:processedFrame];
}

- (UIInterfaceOrientationMask)supportedOverlayOrientationsMask {
    UIInterfaceOrientationMask supportedOverlayOrientationsMask = UIInterfaceOrientationMaskPortrait;
    
    return supportedOverlayOrientationsMask;
}

- (BOOL)isSupportedOverlayOrientation:(UIInterfaceOrientation)orientation {
    return (([self supportedOverlayOrientationsMask] & (1 << orientation)) != 0);
}

- (UIInterfaceOrientation)defaultSupportedOverlayOrientation {
    UIInterfaceOrientation defaultOrientation = (UIInterfaceOrientation)UIDeviceOrientationUnknown;
    UIInterfaceOrientationMask supportedOverlayOrientationsMask = [self supportedOverlayOrientationsMask];
    for (UIInterfaceOrientationMask orientation = UIInterfaceOrientationPortrait;
         orientation <= UIInterfaceOrientationLandscapeRight;
         orientation++) {
        if ((supportedOverlayOrientationsMask & (1 << orientation)) != 0) {
            defaultOrientation = (UIInterfaceOrientation)orientation;
            break;
        }
    }
    return defaultOrientation;
}

@end
