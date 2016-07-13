//
//  ViewController.m
//  NHOCRScanPro
//
//  Created by hu jiaju on 15/7/30.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "ViewController.h"
#import "NHScanVCR.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    self.title = @"OCR Pro";
    
    CGRect infoRect = CGRectMake(100, 100, 100, 50);
    UIButton *btn = [UIButton buttonWithType:UIButtonTypeCustom];
    btn.frame = infoRect;
    [btn setTitle:@"Scan ID" forState:UIControlStateNormal];
    [btn setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
    [btn addTarget:self action:@selector(scanUserID) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:btn];
}

-(void)scanUserID{
    NHScanVCR *scanView = [[NHScanVCR alloc] init];
    [self.navigationController pushViewController:scanView animated:true];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
