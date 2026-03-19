"""
Binary format definitions for .impak files.

File layout:
  [FileHeader]
  [FrameIndexEntry × frame_count]   ← seekable index
  [FrameBlock × frame_count]        ← actual data
    each FrameBlock: [FrameHeader][PatchHeader+Data × patch_count][metadata bytes]

All integers are little-endian.
"""

import struct

MAGIC = b"IMPAK\x00"
VERSION = 1

MODE_VS_FIRST    = 0
MODE_VS_PRIOR    = 1
MODE_KEYFRAME    = 2
MODE_LTO         = 3
MODE_MANUAL      = 4

MODE_NAMES = {
    MODE_VS_FIRST: "vs_first",
    MODE_VS_PRIOR: "vs_prior",
    MODE_KEYFRAME: "keyframe",
    MODE_LTO:      "lto",
    MODE_MANUAL:   "manual",
}
MODE_FROM_NAME = {v: k for k, v in MODE_NAMES.items()}

FRAME_KEYFRAME = 0
FRAME_DELTA    = 1

CODEC_PNG  = 0
CODEC_WEBP = 1

CODEC_NAMES   = {CODEC_PNG: "png", CODEC_WEBP: "webp"}
CODEC_FROM_NAME = {v: k for k, v in CODEC_NAMES.items()}

# FileHeader  (fixed 32 bytes)
#   6s  magic
#   B   version
#   B   diff_mode
#   I   frame_count
#   Q   index_offset
#   I   canvas_width
#   I   canvas_height
#   B   codec           CODEC_PNG or CODEC_WEBP
#   B   quality         0-100
FILE_HEADER_FMT  = "<6sBBIQIIBB"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FMT)

# FrameIndexEntry
#   Q   data_offset
#   I   patch_count
#   I   ref_frame_id
#   I   metadata_len    bytes of JSON metadata appended after patches (0 = none)
#   I   frame_type      FRAME_KEYFRAME or FRAME_DELTA
FRAME_INDEX_ENTRY_FMT  = "<QIIII"
FRAME_INDEX_ENTRY_SIZE = struct.calcsize(FRAME_INDEX_ENTRY_FMT)

PATCH_HEADER_FMT  = "<HHHHIxx"
PATCH_HEADER_SIZE = struct.calcsize(PATCH_HEADER_FMT)


def pack_file_header(diff_mode, frame_count, index_offset, width, height, codec=0, quality=100):
    return struct.pack(
        FILE_HEADER_FMT,
        MAGIC, VERSION, diff_mode,
        frame_count, index_offset,
        width, height,
        codec, quality,
    )


def unpack_file_header(data):
    fields = struct.unpack(FILE_HEADER_FMT, data[:FILE_HEADER_SIZE])
    magic, version, diff_mode, frame_count, index_offset, width, height, codec, quality = fields
    return {
        "magic": magic,
        "version": version,
        "diff_mode": diff_mode,
        "frame_count": frame_count,
        "index_offset": index_offset,
        "width": width,
        "height": height,
        "codec": codec,
        "quality": quality,
    }


def pack_index_entry(data_offset, patch_count, ref_frame_id, metadata_len, frame_type):
    return struct.pack(
        FRAME_INDEX_ENTRY_FMT,
        data_offset, patch_count, ref_frame_id, metadata_len, frame_type,
    )


def unpack_index_entry(data):
    fields = struct.unpack(FRAME_INDEX_ENTRY_FMT, data[:FRAME_INDEX_ENTRY_SIZE])
    data_offset, patch_count, ref_frame_id, metadata_len, frame_type = fields
    return {
        "data_offset": data_offset,
        "patch_count": patch_count,
        "ref_frame_id": ref_frame_id,
        "metadata_len": metadata_len,
        "frame_type": frame_type,
    }


def pack_patch_header(x, y, w, h, data_len):
    return struct.pack(PATCH_HEADER_FMT, x, y, w, h, data_len)


def unpack_patch_header(data):
    x, y, w, h, data_len = struct.unpack(PATCH_HEADER_FMT, data[:PATCH_HEADER_SIZE])
    return x, y, w, h, data_len
