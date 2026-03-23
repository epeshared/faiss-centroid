/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/amx_bf16_block16.h>

#if defined(FAISS_ENABLE_HNSWSQ_BLOCK_AMX)

#include <cstring>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <immintrin.h>

#include <faiss/utils/bf16.h>

namespace faiss::scalar_quantizer::detail {

namespace {

#if defined(__linux__)
constexpr unsigned long kArchGetXCompPerm = 0x1022;
constexpr unsigned long kArchReqXCompPerm = 0x1023;
constexpr unsigned long kXFeatureXTileData = 18;
constexpr unsigned long kXFeatureMaskXTileData = 1UL << kXFeatureXTileData;
#endif

bool enable_amx_for_thread_once() {
#if defined(__linux__)
    static thread_local int state = 0;
    if (state != 0) {
        return state > 0;
    }

    unsigned long bitmask = 0;
    long status = syscall(
            SYS_arch_prctl,
            kArchGetXCompPerm,
            reinterpret_cast<unsigned long>(&bitmask));
    if (status == 0 && (bitmask & kXFeatureMaskXTileData)) {
        state = 1;
        return true;
    }

    status = syscall(SYS_arch_prctl, kArchReqXCompPerm, kXFeatureXTileData);
    if (status != 0) {
        state = -1;
        return false;
    }

    bitmask = 0;
    status = syscall(
            SYS_arch_prctl,
            kArchGetXCompPerm,
            reinterpret_cast<unsigned long>(&bitmask));
    state = (status == 0 && (bitmask & kXFeatureMaskXTileData)) ? 1 : -1;
    return state > 0;
#else
    return false;
#endif
}

float dot_bf16_tail_scalar(const uint16_t* a, const uint16_t* b, uint32_t count) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        acc += decode_bf16(a[i]) * decode_bf16(b[i]);
    }
    return acc;
}

} // namespace

int hnswsq_bf16_amx_batch_x16_single_query(
        const uint16_t* library_bf16,
        const uint16_t* query_bf16,
        size_t dim,
        size_t row_count,
        float* out) {
#if !defined(__AMX_TILE__) || !defined(__AMX_BF16__) || !defined(__AVX512BF16__)
    (void)library_bf16;
    (void)query_bf16;
    (void)dim;
    (void)row_count;
    (void)out;
    return -1;
#else
    if (row_count == 0 || row_count > 16) {
        return -1;
    }
    if (!enable_amx_for_thread_once()) {
        return -1;
    }

    alignas(64) static thread_local unsigned char cfg[64];
    static thread_local int prev_rows = -1;
    const int rows = static_cast<int>(row_count);
    if (prev_rows != rows) {
        std::memset(cfg, 0, sizeof(cfg));
        cfg[0] = 1;

        cfg[16] = 64;
        cfg[48] = static_cast<unsigned char>(rows);

        cfg[18] = 4;
        cfg[49] = 16;

        cfg[20] = 4;
        cfg[50] = static_cast<unsigned char>(rows);

        cfg[22] = 64;
        cfg[51] = static_cast<unsigned char>(rows);
        cfg[24] = 4;
        cfg[52] = 16;

        cfg[26] = 64;
        cfg[53] = static_cast<unsigned char>(rows);
        cfg[28] = 4;
        cfg[54] = 16;

        _tile_loadconfig(cfg);
        prev_rows = rows;
    }

    constexpr int kBlock = 32;
    const size_t block_count = dim / kBlock;
    const size_t tail_count = dim % kBlock;
    const int library_stride_bytes = static_cast<int>(dim * sizeof(uint16_t));
    const char* library = reinterpret_cast<const char*>(library_bf16);
    const char* query = reinterpret_cast<const char*>(query_bf16);

    _tile_zero(2);

    size_t block = 0;
    for (; block + 2 < block_count; block += 3) {
        _tile_loadd(0, library + (block + 0) * kBlock * sizeof(uint16_t), library_stride_bytes);
        _tile_loadd(1, query + (block + 0) * kBlock * sizeof(uint16_t), 4);
        _tile_loadd(3, library + (block + 1) * kBlock * sizeof(uint16_t), library_stride_bytes);
        _tile_loadd(4, query + (block + 1) * kBlock * sizeof(uint16_t), 4);
        _tile_loadd(5, library + (block + 2) * kBlock * sizeof(uint16_t), library_stride_bytes);
        _tile_loadd(6, query + (block + 2) * kBlock * sizeof(uint16_t), 4);

        _tile_dpbf16ps(2, 0, 1);
        _tile_dpbf16ps(2, 3, 4);
        _tile_dpbf16ps(2, 5, 6);
    }

    for (; block < block_count; ++block) {
        _tile_loadd(0, library + block * kBlock * sizeof(uint16_t), library_stride_bytes);
        _tile_loadd(1, query + block * kBlock * sizeof(uint16_t), 4);
        _tile_dpbf16ps(2, 0, 1);
    }

    _tile_stored(2, out, 4);
    _tile_zero(2);

    if (tail_count != 0) {
        const size_t base = block_count * kBlock;
        for (size_t row = 0; row < row_count; ++row) {
            out[row] += dot_bf16_tail_scalar(
                    library_bf16 + row * dim + base,
                    query_bf16 + base,
                    static_cast<uint32_t>(tail_count));
        }
    }

    return 0;
#endif
}

int hnswsq_bf16_amx_batch_queries_x16(
        const uint16_t* library_bf16,
        const uint16_t* queries_bf16,
        size_t dim,
        size_t row_count,
        size_t query_count,
        float* out) {
    if (query_count == 0) {
        return 0;
    }

    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        int status = hnswsq_bf16_amx_batch_x16_single_query(
                library_bf16,
                queries_bf16 + query_id * dim,
                dim,
                row_count,
                out + query_id * row_count);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

} // namespace faiss::scalar_quantizer::detail

#else

namespace faiss::scalar_quantizer::detail {

int hnswsq_bf16_amx_batch_x16_single_query(
        const uint16_t* library_bf16,
        const uint16_t* query_bf16,
        size_t dim,
        size_t row_count,
        float* out) {
    (void)library_bf16;
    (void)query_bf16;
    (void)dim;
    (void)row_count;
    (void)out;
    return -1;
}

int hnswsq_bf16_amx_batch_queries_x16(
        const uint16_t* library_bf16,
        const uint16_t* queries_bf16,
        size_t dim,
        size_t row_count,
        size_t query_count,
        float* out) {
    (void)library_bf16;
    (void)queries_bf16;
    (void)dim;
    (void)row_count;
    (void)query_count;
    (void)out;
    return -1;
}

} // namespace faiss::scalar_quantizer::detail

#endif
