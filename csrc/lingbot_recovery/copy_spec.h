// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
struct CopySpec {
    int64_t sizes[5];
    int64_t src_strides[5];
    int64_t dst_strides[5];
    int64_t rows;
    int64_t width;
};
