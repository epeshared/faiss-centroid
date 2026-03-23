# Build Commands for AMX and AVX512_SPR

This document records the CMake configuration and build commands used in this workspace to build `faiss-centroid` for:

- `amx`
- `avx512_spr`

The examples below build CPU-only Release binaries and disable tests.

## Common Build Configuration

Shared CMake options used by both builds:

```bash
-DFAISS_ENABLE_GPU=OFF
-DBUILD_TESTING=OFF
-DCMAKE_BUILD_TYPE=Release
```

If you also need Python bindings, keep `FAISS_ENABLE_PYTHON` at its default `ON` and build the corresponding `swigfaiss_*` target.

## 1. Build AMX

### Configure

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

cmake -S . -B build-split-amx \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=amx
```

### Build C++ library target

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

cmake --build build-split-amx -j4 --target faiss_amx
```

### Build Python bindings target

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

cmake --build build-split-amx -j8 --target swigfaiss_amx
```

### Output locations

- C++ library artifacts are generated under `build-split-amx/faiss/`
- Python binding artifacts are generated under `build-split-amx/faiss/python/`

## 2. Build AVX512_SPR

### Configure

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

cmake -S . -B build-split-avx512spr \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx512_spr
```

### Build C++ library target

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

cmake --build build-split-avx512spr -j4 --target faiss_avx512_spr
```

### Build Python bindings target

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

cmake --build build-split-avx512spr -j8 --target swigfaiss_avx512_spr
```

### Output locations

- C++ library artifacts are generated under `build-split-avx512spr/faiss/`
- Python binding artifacts are generated under `build-split-avx512spr/faiss/python/`

## 3. Optional Clean Rebuild

If you want to rebuild from scratch, remove the corresponding build directory first:

```bash
cd /nvme5/xtang/vdb-workspace/faiss-centroid

rm -rf build-split-amx
rm -rf build-split-avx512spr
```

Then rerun the configure and build commands above.

## 4. Notes

- `FAISS_OPT_LEVEL=amx` produces the `faiss_amx` and `swigfaiss_amx` targets.
- `FAISS_OPT_LEVEL=avx512_spr` produces the `faiss_avx512_spr` and `swigfaiss_avx512_spr` targets.
- These commands match the build flow already used in this workspace.
- If you only need the C++ library, you do not need to build the `swigfaiss_*` target.