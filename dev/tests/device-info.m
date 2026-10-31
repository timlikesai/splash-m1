// What GPU is this, in the terms the runtime chooses kernels by.
//
// system_profiler reports nothing on a headless VM and `hw.model` names the
// virtual machine, so the only honest answer comes from Metal itself. CI prints
// this before the tests because a failure means something different on a family
// 7 slice than on a family 9 one, and because a hosted runner with no ANE or no
// residency set cannot run part of this suite at all.
//
//   cc -framework Metal -framework Foundation -o device-info device-info.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString *familyName(MTLGPUFamily family) {
  switch (family) {
    case MTLGPUFamilyApple3: return @"Apple3";
    case MTLGPUFamilyApple4: return @"Apple4";
    case MTLGPUFamilyApple5: return @"Apple5";
    case MTLGPUFamilyApple6: return @"Apple6";
    case MTLGPUFamilyApple7: return @"Apple7";
    case MTLGPUFamilyApple8: return @"Apple8";
    case MTLGPUFamilyApple9: return @"Apple9";
    case MTLGPUFamilyApple10: return @"Apple10";
    case MTLGPUFamilyApple11: return @"Apple11";
    case MTLGPUFamilyMetal4: return @"Metal4";
    default: return @"?";
  }
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stdout, "Metal: no device\n");
      return 1;
    }
    NSMutableString *families = [NSMutableString string];
    for (MTLGPUFamily family = MTLGPUFamilyApple3; family <= MTLGPUFamilyApple11; ++family)
      if ([device supportsFamily:family])
        [families appendFormat:@"%s ", familyName(family).UTF8String];
    // The two things this branch's code asks about outside the family list.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    BOOL residency = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                                    error:nil] != nil;
    fprintf(stdout, "%s: %s GB working set, %s%s%s\n",
            device.name.UTF8String,
            [@(device.recommendedMaxWorkingSetSize / (1ULL << 30)) stringValue].UTF8String,
            families.UTF8String,
            device.hasUnifiedMemory ? ", unified" : ", NOT unified",
            residency ? ", residency sets" : ", NO residency sets");
    // Nothing is released: this runs once, prints, and exits. Under ARC the
    // message is unavailable, and a probe that only builds one way is a probe
    // that gets edited out of a log the next time someone needs it.
    return 0;
  }
}
