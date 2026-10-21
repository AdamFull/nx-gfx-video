#pragma once

#include "core/foundation/containers/small_vector.h"
#include "nx/gfx/types.h"
#include "nx/gfx/vulkan.h"

namespace nx::gfx::video {

enum class video_codec : u8 { VP9, AV1, Count };

struct decode_caps {
  bool supported = false;
  u32 max_width = 0;
  u32 max_height = 0;
  u32 max_dpb_slots = 0;
  bool dpb_and_output_coincide = false;
};

[[nodiscard]] decode_caps query_decode_caps(const nx::gfx::device &device,
                                            video_codec codec);

/// The queue family of @p adapter that decodes VP9 and draws nothing, or ~0u
/// where the device has none or lacks the extensions and the feature.
[[nodiscard]] u32 find_vp9_decode_family(VkPhysicalDevice adapter);

struct video_image {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct decoded_picture {
  bool ok = false;
  u32 slot = 0;
  bool show = false;
  u32 width = 0;
  u32 height = 0;
};

class video_decoder {
public:
  video_decoder() = default;
  ~video_decoder();

  video_decoder(const video_decoder &) = delete;
  video_decoder &operator=(const video_decoder &) = delete;

  /// Decodes on the queue @p device made for video_decode_client(); false
  /// where it made none.
  [[nodiscard]] bool init(nx::gfx::device &device, video_codec codec, u32 width,
                          u32 height);
  void shutdown() noexcept;

  [[nodiscard]] bool ready() const noexcept {
    return m_session != VK_NULL_HANDLE;
  }
  [[nodiscard]] VkExtent2D extent() const noexcept { return m_extent; }
  [[nodiscard]] u32 dpb_slots() const noexcept { return m_max_dpb_slots; }

  [[nodiscard]] u32 dpb_image_count() const noexcept {
    return nx::cast<u32>(m_dpb.size());
  }
  [[nodiscard]] VkImageView dpb_view(u32 slot) const noexcept {
    return slot < m_dpb.size() ? m_dpb[slot].view : VK_NULL_HANDLE;
  }

  [[nodiscard]] bool decode_frame(const u8 *frame, u32 size, bool &shown,
                                  u32 &width, u32 &height);
  /// Copies the last decoded picture into new luma and chroma textures and
  /// retires the previous ones. False, keeping them, while every plane pair
  /// is still read by frames in flight.
  [[nodiscard]] bool show_last();
  [[nodiscard]] nx::gfx::texture_handle luma_texture() const noexcept {
    return m_y;
  }
  [[nodiscard]] nx::gfx::texture_handle chroma_texture() const noexcept {
    return m_cbcr;
  }
  [[nodiscard]] nx::gfx::extent2d coded_extent() const noexcept {
    return {m_extent.width, m_extent.height};
  }

  void reset_stream() noexcept;

private:
  [[nodiscard]] bool decode(const u8 *frame, u32 size, decoded_picture &out);
  [[nodiscard]] bool present(u32 slot);
  [[nodiscard]] bool create_command_resources();
  /// The submission recorded next, begun once the queue is done with it; ~0u
  /// on failure.
  [[nodiscard]] u32 begin_submission();
  /// Submits it, signalling the timeline's next value.
  [[nodiscard]] bool end_submission(u32 index);
  [[nodiscard]] bool
  create_bitstream_buffer(u64 capacity,
                          const VkVideoProfileListInfoKHR &profiles);
  [[nodiscard]] u32 allocate_slot() noexcept;

  struct plane_pair;
  struct plane_pool;
  [[nodiscard]] plane_pair *free_planes();
  [[nodiscard]] nx::gfx::texture_handle
  wrap_plane(plane_pair &pair, const video_image &plane, nx::gfx::format format,
             VkExtent2D extent, nx::string_view name);

  nx::gfx::device *m_device = nullptr;
  VkQueue m_video_queue = VK_NULL_HANDLE;
  u32 m_video_family = ~0u;
  VkVideoSessionKHR m_session = VK_NULL_HANDLE;
  nx::small_vector<VkDeviceMemory, 4> m_memories;
  nx::small_vector<video_image, 16> m_dpb;
  video_image m_output;
  bool m_coincide = false;
  VkFormat m_picture_format = VK_FORMAT_UNDEFINED;
  VkFormat m_dpb_format = VK_FORMAT_UNDEFINED;
  VkExtent2D m_extent = {};
  VkExtent2D m_coded_limit = {};
  u32 m_max_dpb_slots = 0;

  VkCommandPool m_pool = VK_NULL_HANDLE;
  static constexpr u32 SUBMISSIONS = 4;
  struct submission {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    /// The timeline value its last submit signals.
    u64 value = 0;
  };
  submission m_submissions[SUBMISSIONS];
  u32 m_next_submission = 0;
  /// Every submission signals it, and frames wait on it for the planes; the
  /// device owns it.
  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_submitted = 0;
  bool m_video_supports_transfer = false;
  bool m_needs_reset = true;

  VkBuffer m_bitstream = VK_NULL_HANDLE;
  VkDeviceMemory m_bitstream_memory = VK_NULL_HANDLE;
  void *m_bitstream_mapped = nullptr;
  /// Per submission, each its own range of the buffer.
  u64 m_bitstream_capacity = 0;
  u64 m_bitstream_stride = 0;
  u64 m_bitstream_offset_align = 1;
  u64 m_bitstream_size_align = 1;

  u32 m_vp9_to_dpb[8] = {};
  bool m_vp9_valid[8] = {};
  u32 m_slot_width[16] = {};
  u32 m_slot_height[16] = {};
  u32 m_next_slot = 0;
  u32 m_last_slot = 0;
  bool m_have_last = false;

  /// Shown pictures go to a free pair of plane images, shared with the
  /// device's queue families, which the device samples as wrapped textures.
  plane_pool *m_planes = nullptr;
  nx::small_vector<u32, 4> m_plane_families;
  nx::gfx::texture_handle m_y;
  nx::gfx::texture_handle m_cbcr;
};

} // namespace nx::gfx::video
