//
//  NHGPUTransformFilter.m
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#import "NHGPUTransformFilter.h"
#import "NHGPUShaders.h"

NSString *const kTransformVertexShader = SHADER_STRING
(
 attribute vec4 position;    // input position
 attribute vec2 texCoordIn;  // input texture coordinate
 varying   vec2 texCoordOut; // output texture coordinate (goes to frag shader)
 // "varying" means OpenGL will use interpolation to
 // determine color
 
 uniform mat4 transformMatrix;
 uniform mat4 orthographicMatrix;
 
 void main(void) {
     gl_Position = vec4(orthographicMatrix * transformMatrix * position);
     texCoordOut = texCoordIn;
 }
 );

// Fragment shader for drawing texture
NSString *const kPassthroughFragmentShader = SHADER_STRING
(
 varying lowp vec2 texCoordOut; // input texture coordinate (from vertex shader)
 uniform sampler2D texture; // input texture
 
 void main(void) {
     gl_FragColor = texture2D(texture, texCoordOut).rgba; // Interpolate texture for texCoordOut
 }
 );


@interface NHGPUTransformFilter ()

@end

@implementation NHGPUTransformFilter

- (id)initWithSize:(CGSize)size {
    if((self = [super initWithSize:size vertexShaderSrc:kTransformVertexShader fragmentShaderSrc:kPassthroughFragmentShader])) {
        [_gpuRenderer withContextDo:^{
            // Get our matrix handles
            _transformMatrixUniform = [_gpuRenderer uniformIndex:@"transformMatrix"];
            _orthographicMatrixUniform = [_gpuRenderer uniformIndex:@"orthographicMatrix"];
            
            // Set up the ortho matrix
            // Could hard-code this into the shader. But it's easier to understand in this form, I think.
            [[self class] loadOrthoMatrix:orthographicMatrix left:-1.0 right:1.0 bottom:-1.0 top:1.0 near:-1.0 far:1.0];
            glUniformMatrix4fv(_orthographicMatrixUniform, 1, GL_FALSE, orthographicMatrix);
        }];
    }
    return self;
}

// Sets the matrix data for use with the vertex shader
- (void)setPerspectiveMat:(float *)matrix {
    [_gpuRenderer withContextDo:^{
        [_gpuRenderer prepareForUse];
        //NSLog(@"GL perspective matrix: \n%@", [[self class] matrixAsString:matrix size:4 rowMajor:NO]);
        glUniformMatrix4fv(_transformMatrixUniform, 1, GL_FALSE, matrix);
    }];
}

@end
