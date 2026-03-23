/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss::scalar_quantizer::detail {

int hnswsq_bf16_amx_batch_x16_single_query(
        const uint16_t* library_bf16,
        const uint16_t* query_bf16,
        size_t dim,
        size_t row_count,
        float* out);

int hnswsq_bf16_amx_batch_queries_x16(
        const uint16_t* library_bf16,
        const uint16_t* queries_bf16,
        size_t dim,
        size_t row_count,
        size_t query_count,
        float* out);

} // namespace faiss::scalar_quantizer::detail
