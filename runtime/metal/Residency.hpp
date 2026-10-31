#pragma once

#include "AwakeClock.hpp"
#include "MetalBackend.hpp"

#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string_view>

namespace splash::metal {

// Holds every buffer of a backend in one residency set, wired between the
// commands of its command queue. One request holds them only about two
// seconds, so a heartbeat requests residency every 500 ms. After keepAlive
// without a command, never when it is infinite, it ends residency, which Metal
// applies at its next GPU operation on any queue: one dispatch of the kick
// kernel, whose pipeline the backend builds with its library, on a queue of
// its own, as the runtime keeps exactly one command in flight on the command
// queue. Destruction ends residency without GPU work: a backend being torn
// down must not start any. The sets are used only on the heartbeat's serial
// queue.
//
// The command queue holds a set of its own, of the buffers kernels reach only
// through their GPU addresses (BufferReach::Addressed): Metal makes it
// resident for every command buffer, as it does each buffer a dispatch binds.
// Holding every buffer there would make each command buffer wait for all of
// them first: with memory nearly full, 1-2% of command buffers waited
// 100-170 ms before they started (M5 Max with 36 GB), so a command of many
// buffers waited at nearly every command.
class Residency final {
public:
  // A one-thread kernel that writes one word of its buffer 0
  // (kernels/shared/residency.metal).
  static constexpr std::string_view kKickPipeline = "residency_kick";

  // A device without residency sets (a virtualized GPU, where
  // newResidencySetWithDescriptor returns nil) gets an object that does
  // nothing: Metal wires each buffer for the command buffers that name it, as
  // it does when no set holds it. Callers cannot tell the difference, and the
  // backend says so once at startup, because host memory then moves by the
  // weight size while a request runs (MetalBackend::residencyAvailable).
  Residency(id<MTLDevice> device, id<MTLCommandQueue> commands,
            id<MTLComputePipelineState> kick, double keepAliveSeconds)
      : commands_(commands), kick_(kick), keepAlive_(keepAliveSeconds) {
    set_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                           error:nil];
    addressed_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                                 error:nil];
    if (!set_ || !addressed_) return;
    kickQueue_ = [device newCommandQueue];
    kickTarget_ = [device newBufferWithLength:sizeof(uint32_t)
                                      options:MTLResourceStorageModePrivate];
    if (!kickQueue_ || !kickTarget_)
      throw MetalBackendError("unable to create the Metal residency kick");
    available_ = true;
    [commands_ addResidencySet:addressed_];
    queue_ = dispatch_queue_create(
        "splash.metal.residency",
        dispatch_queue_attr_make_with_autorelease_frequency(
            DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM));
    dispatch_source_set_event_handler(heartbeat_, ^{ beat(); });
    dispatch_activate(heartbeat_);
  }

  ~Residency() {
    if (!available_) return;
    dispatch_source_cancel(heartbeat_);
    // Runs after any beat or request already queued: they use this object.
    // Metal applies the end at the process's next GPU operation, or frees
    // the set at exit.
    dispatch_sync(queue_, ^{
      [commands_ removeResidencySet:addressed_];
      std::lock_guard lock(mutex_);
      if (held_) [set_ endResidency];
    });
  }

  // Whether this device had residency sets at all.
  [[nodiscard]] bool available() const noexcept { return available_; }

  Residency(const Residency &) = delete;
  Residency &operator=(const Residency &) = delete;

  // Adds the buffer to the set, and an addressed one to the queue's too, and
  // restarts the keep-alive: a held set wires it at the commit, and a lapsed
  // one is requested again off the caller's thread (use()).
  void add(id<MTLBuffer> buffer, BufferReach reach) {
    if (!available_) return;
    dispatch_sync(queue_, ^{
      [set_ addAllocation:buffer];
      [set_ commit];
      if (reach == BufferReach::Addressed) {
        [addressed_ addAllocation:buffer];
        [addressed_ commit];
      }
    });
    use();
  }

  void remove(id<MTLBuffer> buffer, BufferReach reach) {
    if (!available_) return;
    dispatch_sync(queue_, ^{
      [set_ removeAllocation:buffer];
      [set_ commit];
      if (reach == BufferReach::Addressed) {
        [addressed_ removeAllocation:buffer];
        [addressed_ commit];
      }
    });
  }

  // Marks a command. A lapsed set is requested again at once, off the
  // caller's thread, and the heartbeat resumes.
  void use() {
    if (!available_) return;
    {
      std::lock_guard lock(mutex_);
      lastUse_ = AwakeClock::now();
      if (held_) return;
      held_ = true;
    }
    dispatch_async(queue_, ^{
      [set_ requestResidency];
      dispatch_source_set_timer(heartbeat_, dispatch_time(DISPATCH_TIME_NOW, kBeat),
                                kBeat, kBeat / 10);
    });
  }

private:
  static constexpr uint64_t kBeat = 500 * NSEC_PER_MSEC;

  void beat() {
    bool lapsed;
    {
      std::lock_guard lock(mutex_);
      lapsed = AwakeClock::now() - lastUse_ >= keepAlive_;
      held_ = !lapsed;
    }
    if (!lapsed) {
      [set_ requestResidency];
      return;
    }
    dispatch_source_set_timer(heartbeat_, DISPATCH_TIME_FOREVER, 0, 0);
    [set_ endResidency];
    kickResidencyEnd();
  }

  void kickResidencyEnd() {
    id<MTLCommandBuffer> command = [kickQueue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:kick_];
    [encoder setBuffer:kickTarget_ offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
  }

  __strong id<MTLCommandQueue> commands_ = nil;
  __strong id<MTLComputePipelineState> kick_ = nil;
  __strong id<MTLResidencySet> set_ = nil;
  // The command queue's: the addressed buffers.
  __strong id<MTLResidencySet> addressed_ = nil;
  __strong id<MTLCommandQueue> kickQueue_ = nil;
  __strong id<MTLBuffer> kickTarget_ = nil;
  __strong dispatch_queue_t queue_ = nil;
  __strong dispatch_source_t heartbeat_ = nil;
  const std::chrono::duration<double> keepAlive_;
  std::mutex mutex_;
  AwakeClock::time_point lastUse_;
  bool held_ = false;
  // False on a device with no residency sets, where everything here is inert.
  bool available_ = false;
};

} // namespace splash::metal
