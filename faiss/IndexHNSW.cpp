/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>

#include <cstdint>
#include "faiss/Index.h"

#include <faiss/Index2Layer.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/scalar_quantizer/amx_bf16_block16.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++) {
        omp_init_lock(&locks[i]);
    }

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size()) {
                hist.push_back(0);
            }
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1;
             pt_level >= int(!index_hnsw.init_level0);
             pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++) {
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
            }

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal, hnsw.use_visited_hashset);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hnsw.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0));

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        if (index_hnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
        : Index(d, metric), hnsw(M) {}

IndexHNSW::IndexHNSW(Index* storage, int M)
        : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {
    metric_arg = storage->metric_arg;
}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void hnsw_search(
        const IndexHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    const HNSW& hnsw = index->hnsw;

    int efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            efSearch = hnsw_params->efSearch;
        }
    }
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal, hnsw.use_visited_hashset);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HNSWStats stats = hnsw.search(*dis, index, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

} // anonymous namespace

void IndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::search1(
        const float* x,
        ResultHandler& handler,
        SearchParameters* params) const {
    SingleQueryBlockResultHandler<HNSW::C, false> bres(handler);
    hnsw_search(this, 1, x, bres, params);
}

void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}

void IndexHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

/**************************************************************
 * This section of functions were used during the development of HNSW support.
 * They may be useful in the future but are dormant for now, and thus are not
 * unit tested at the moment.
 * shrink_level_0_neighbors
 * search_level_0
 * init_level_0_from_knngraph
 * init_level_0_from_entry_points
 * reorder_links
 * link_singletons
 **************************************************************/
void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0) {
                    break;
                }
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size()) {
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                } else {
                    hnsw.neighbors[j] = -1;
                }
            }
        }
    }
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    storage_idx_t ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(
                storage_distance_computer(storage));
        HNSWStats search_stats;
        VisitedTable vt(ntotal, hnsw.use_visited_hashset);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            hnsw.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt,
                    params);
            res.end();
            vt.advance();
        }
#pragma omp critical
        {
            hnsw_stats.combine(search_stats);
        }
    }
    if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
        for (int64_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i) {
                continue;
            }
            if (v1 < 0) {
                break;
            }
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size()) {
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            } else {
                hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSW::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++) {
        omp_init_lock(&locks[i]);
    }

#pragma omp parallel
    {
        VisitedTable vt(ntotal, hnsw.use_visited_hashset);

        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

void IndexHNSW::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSW::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0) {
                seen[ni] = true;
            }
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1) {
                n_sing_l1++;
            }
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(false); // not implemented
    }
}

void IndexHNSW::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    flat_storage->permute_entries(perm);
    hnsw.permute_entries(perm);
}

DistanceComputer* IndexHNSW::get_distance_computer() const {
    return storage->get_distance_computer();
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
    is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWFlatPanorama implementation
 **************************************************************/

IndexHNSWFlatPanorama::IndexHNSWFlatPanorama()
        : IndexHNSWFlat(), cum_sums(), pano(0, 1, 1), num_panorama_levels(0) {}

IndexHNSWFlatPanorama::IndexHNSWFlatPanorama(
        int d,
        int M,
        int num_panorama_levels,
        MetricType metric)
        : IndexHNSWFlat(d, M, metric),
          cum_sums(),
          pano(d * sizeof(float), num_panorama_levels, 1),
          num_panorama_levels(num_panorama_levels) {
    // For now, we only support L2 distance.
    // Supporting dot product and cosine distance is a trivial addition
    // left for future work.
    FAISS_THROW_IF_NOT(metric == METRIC_L2);

    // Enable Panorama search mode.
    // This is not ideal, but is still more simple than making a subclass of
    // HNSW and overriding the search logic.
    hnsw.is_panorama = true;
}

void IndexHNSWFlatPanorama::add(idx_t n, const float* x) {
    idx_t n0 = ntotal;
    cum_sums.resize((ntotal + n) * (pano.n_levels + 1));
    pano.compute_cumulative_sums(cum_sums.data(), n0, n, x);
    IndexHNSWFlat::add(n, x);
}

void IndexHNSWFlatPanorama::reset() {
    cum_sums.clear();
    IndexHNSWFlat::reset();
}

void IndexHNSWFlatPanorama::permute_entries(const idx_t* perm) {
    std::vector<float> new_cum_sums(ntotal * (pano.n_levels + 1));

    for (idx_t i = 0; i < ntotal; i++) {
        idx_t src = perm[i];
        memcpy(&new_cum_sums[i * (pano.n_levels + 1)],
               &cum_sums[src * (pano.n_levels + 1)],
               (pano.n_levels + 1) * sizeof(float));
    }

    std::swap(cum_sums, new_cum_sums);
    IndexHNSWFlat::permute_entries(perm);
}

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

IndexHNSWPQ::IndexHNSWPQ() = default;

IndexHNSWPQ::IndexHNSWPQ(
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        MetricType metric)
        : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHNSWSQ helpers
 **************************************************************/

namespace {

struct HNSWSQBlockSearchConfig {
    bool use_block_centroid_search = false;
    int top_b = -1;
    int centroid_ef_construction = 0;
    int centroid_ef_search = 0;
};

// HNSWSQ BF16 runtime configuration, read once per process from environment.
// This config only affects the optional block-centroid path for
// IndexHNSWSQ + ScalarQuantizer::QT_bf16.
//
// Supported variables:
// - FAISS_HNSWSQ_BF16_USE_BLOCK_CENTROID_SEARCH
//     Enables the optional 16-vector block + centroid-HNSW path.
//     Accepted true values: 1/true/yes/on.
//     Accepted false values: 0/false/no/off.
//     Default: false.
// - FAISS_HNSWSQ_BF16_TOP_B
//     Number of centroid-selected blocks to scan per query.
//     Default: top-k.
// - FAISS_HNSWSQ_BF16_CENTROID_EFC
//     efConstruction used when building the centroid HNSW.
//     Default: reuse the main HNSW efConstruction.
// - FAISS_HNSWSQ_BF16_CENTROID_EFS
//     efSearch used when searching the centroid HNSW.
//     Default: reuse the main HNSW/SearchParametersHNSW efSearch.

bool getenv_bool_once(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }

    if (!strcmp(value, "1") || !strcasecmp(value, "true") ||
        !strcasecmp(value, "yes") || !strcasecmp(value, "on")) {
        return true;
    }
    if (!strcmp(value, "0") || !strcasecmp(value, "false") ||
        !strcasecmp(value, "no") || !strcasecmp(value, "off")) {
        return false;
    }
    return default_value;
}

int getenv_int_once(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }

    char* endptr = nullptr;
    long parsed = std::strtol(value, &endptr, 10);
    if (endptr == value || *endptr != '\0') {
        return default_value;
    }
    if (parsed < std::numeric_limits<int>::min()) {
        return default_value;
    }
    if (parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

const HNSWSQBlockSearchConfig& get_hnswsq_block_search_config() {
    static const HNSWSQBlockSearchConfig config = {
        getenv_bool_once(
            "FAISS_HNSWSQ_BF16_USE_BLOCK_CENTROID_SEARCH",
            false),
            getenv_int_once("FAISS_HNSWSQ_BF16_TOP_B", -1),
            getenv_int_once("FAISS_HNSWSQ_BF16_CENTROID_EFC", 0),
            getenv_int_once("FAISS_HNSWSQ_BF16_CENTROID_EFS", 0)};
    return config;
}

bool is_hnswsq_bf16(const IndexHNSWSQ& index) {
    const auto* sq_storage = dynamic_cast<const IndexScalarQuantizer*>(index.storage);
    return sq_storage &&
            sq_storage->sq.qtype == ScalarQuantizer::QT_bf16;
}

struct ProfilingDistanceComputer : DistanceComputer {
    std::unique_ptr<DistanceComputer> base;
    double distance_compute_time_s = 0.0;

    explicit ProfilingDistanceComputer(std::unique_ptr<DistanceComputer> base)
            : base(std::move(base)) {}

    void set_query(const float* x) override {
        base->set_query(x);
    }

    float operator()(idx_t i) override {
        const double t0 = getmillisecs();
        const float result = (*base)(i);
        distance_compute_time_s += (getmillisecs() - t0) * 1e-3;
        return result;
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        const double t0 = getmillisecs();
        base->distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        distance_compute_time_s += (getmillisecs() - t0) * 1e-3;
    }

    void distances_batch_16(const idx_t* idx, size_t count, float* dis) override {
        const double t0 = getmillisecs();
        base->distances_batch_16(idx, count, dis);
        distance_compute_time_s += (getmillisecs() - t0) * 1e-3;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const double t0 = getmillisecs();
        const float result = base->symmetric_dis(i, j);
        distance_compute_time_s += (getmillisecs() - t0) * 1e-3;
        return result;
    }
};

void clear_hnswsq_search_profile(IndexHNSWSQ& index) {
    index.last_search_profile = HNSWSQSearchProfile();
}

void clear_hnswsq_block_state(IndexHNSWSQ& index) {
    clear_hnswsq_search_profile(index);
    index.block_offsets.clear();
    index.reordered_to_original.clear();
    index.original_to_reordered.clear();
    index.original_database.clear();
    index.block_centroids.clear();
    index.centroid_storage = IndexScalarQuantizer(
            index.d,
            ScalarQuantizer::QT_bf16,
            index.metric_type);
    index.centroid_hnsw = HNSW(index.hnsw.nb_neighbors(1));
}

void rebuild_hnswsq_centroid_storage(IndexHNSWSQ& index) {
    index.centroid_storage = IndexScalarQuantizer(
            index.d,
            ScalarQuantizer::QT_bf16,
            index.metric_type);
    if (!index.block_centroids.empty()) {
        idx_t nblocks = index.block_centroids.size() / index.d;
        index.centroid_storage.add(nblocks, index.block_centroids.data());
    }
}

struct ProposalEntry {
    float distance;
    idx_t block_id;
    idx_t vector_id;
};

void encode_matrix_bf16(
        const float* input,
        idx_t rows,
        idx_t dim,
        std::vector<uint16_t>& output) {
    output.resize(static_cast<size_t>(rows) * dim);
    for (idx_t i = 0; i < rows * dim; ++i) {
        output[i] = encode_bf16(input[i]);
    }
}

void insert_topk_similarity(
        float score,
        idx_t label,
        float* top_scores,
        idx_t* top_labels,
        idx_t k) {
    if (k <= 0 || score <= top_scores[k - 1]) {
        return;
    }

    idx_t pos = k - 1;
    while (pos > 0 && score > top_scores[pos - 1]) {
        top_scores[pos] = top_scores[pos - 1];
        top_labels[pos] = top_labels[pos - 1];
        --pos;
    }
    top_scores[pos] = score;
    top_labels[pos] = label;
}

void insert_topk_distance(
        float score,
        idx_t label,
        float* top_scores,
        idx_t* top_labels,
        idx_t k) {
    if (k <= 0 || score >= top_scores[k - 1]) {
        return;
    }

    idx_t pos = k - 1;
    while (pos > 0 && score < top_scores[pos - 1]) {
        top_scores[pos] = top_scores[pos - 1];
        top_labels[pos] = top_labels[pos - 1];
        --pos;
    }
    top_scores[pos] = score;
    top_labels[pos] = label;
}

void collect_centroid_candidates_bf16(
        const float* x,
        idx_t n,
        idx_t d,
    const float* centroids,
    idx_t nblocks,
    MetricType metric_type,
        idx_t candidate_k,
        std::vector<ProposalEntry>& candidates) {
    const bool use_similarity = metric_type == METRIC_INNER_PRODUCT;
    const bool use_amx_ip = use_similarity;

    std::vector<uint16_t> database_bf16;
    std::vector<uint16_t> centroid_bf16;
    encode_matrix_bf16(x, n, d, database_bf16);
    encode_matrix_bf16(centroids, nblocks, d, centroid_bf16);

    const idx_t query_batch_size = 32;
    candidates.clear();
    candidates.resize(static_cast<size_t>(nblocks) * candidate_k);

#pragma omp parallel for schedule(dynamic)
    for (idx_t centroid_begin = 0; centroid_begin < nblocks;
         centroid_begin += query_batch_size) {
        const idx_t query_count =
                std::min<idx_t>(query_batch_size, nblocks - centroid_begin);
        std::vector<float> top_scores(query_count * candidate_k);
        std::vector<idx_t> top_labels(query_count * candidate_k, -1);

        if (use_similarity) {
            std::fill(
                    top_scores.begin(),
                    top_scores.end(),
                    -std::numeric_limits<float>::infinity());
        } else {
            std::fill(
                    top_scores.begin(),
                    top_scores.end(),
                    std::numeric_limits<float>::infinity());
        }

        std::vector<float> block_scores(query_count * 16);

        for (idx_t database_begin = 0; database_begin < n; database_begin += 16) {
            const idx_t row_count = std::min<idx_t>(16, n - database_begin);
            bool used_amx = false;

            if (use_amx_ip) {
                used_amx =
                        scalar_quantizer::detail::hnswsq_bf16_amx_batch_queries_x16(
                                database_bf16.data() + database_begin * d,
                                centroid_bf16.data() + centroid_begin * d,
                                static_cast<size_t>(d),
                                static_cast<size_t>(row_count),
                                static_cast<size_t>(query_count),
                                block_scores.data()) == 0;
            }

            for (idx_t local_query = 0; local_query < query_count; ++local_query) {
                const float* query = centroids + (centroid_begin + local_query) * d;
                float* query_top_scores =
                        top_scores.data() + local_query * candidate_k;
                idx_t* query_top_labels =
                        top_labels.data() + local_query * candidate_k;

                for (idx_t local_row = 0; local_row < row_count; ++local_row) {
                    const idx_t vector_id = database_begin + local_row;
                    float score = 0.0f;
                    if (used_amx) {
                        score =
                                block_scores[local_query * row_count + local_row];
                    } else if (use_similarity) {
                        const uint16_t* query_bf16 =
                                centroid_bf16.data() + (centroid_begin + local_query) * d;
                        const uint16_t* vector_bf16 =
                                database_bf16.data() + vector_id * d;
                        for (idx_t dim = 0; dim < d; ++dim) {
                            score += decode_bf16(query_bf16[dim]) *
                                    decode_bf16(vector_bf16[dim]);
                        }
                    } else {
                        const float* vector = x + vector_id * d;
                        for (idx_t dim = 0; dim < d; ++dim) {
                            const float diff = query[dim] - vector[dim];
                            score += diff * diff;
                        }
                    }

                    if (use_similarity) {
                        insert_topk_similarity(
                                score,
                                vector_id,
                                query_top_scores,
                                query_top_labels,
                                candidate_k);
                    } else {
                        insert_topk_distance(
                                score,
                                vector_id,
                                query_top_scores,
                                query_top_labels,
                                candidate_k);
                    }
                }
            }
        }

        for (idx_t local_query = 0; local_query < query_count; ++local_query) {
            const idx_t block_id = centroid_begin + local_query;
            for (idx_t rank = 0; rank < candidate_k; ++rank) {
                const size_t local_offset =
                        static_cast<size_t>(local_query) * candidate_k + rank;
                const size_t global_offset =
                        static_cast<size_t>(block_id) * candidate_k + rank;
                candidates[global_offset] = {
                        top_scores[local_offset], block_id, top_labels[local_offset]};
            }
        }
    }
}

void collect_block_members(
        const float* x,
        idx_t n,
        idx_t d,
    const float* centroids,
    idx_t nblocks,
    MetricType metric_type,
        idx_t block_size,
        std::vector<idx_t>& block_members) {
    FAISS_THROW_IF_NOT(nblocks > 0);

    std::vector<ProposalEntry> candidates;
    collect_centroid_candidates_bf16(
            x,
            n,
            d,
        centroids,
        nblocks,
        metric_type,
            block_size,
            candidates);

    block_members.resize(static_cast<size_t>(nblocks) * block_size);
    for (idx_t block_id = 0; block_id < nblocks; ++block_id) {
        for (idx_t rank = 0; rank < block_size; ++rank) {
            const ProposalEntry& candidate =
                    candidates[static_cast<size_t>(block_id) * block_size + rank];
            FAISS_THROW_IF_NOT_FMT(
                    candidate.vector_id >= 0,
                    "block %" PRId64 " top-%" PRId64 " candidate missing",
                    int64_t(block_id),
                    int64_t(rank));
            block_members[static_cast<size_t>(block_id) * block_size + rank] =
                    candidate.vector_id;
        }
    }
}

void rebuild_hnswsq_bf16_blocks(IndexHNSWSQ& index, idx_t n, const float* x) {
    idx_t old_total = index.ntotal;
    idx_t total = old_total + n;
    auto* sq_storage = dynamic_cast<IndexScalarQuantizer*>(index.storage);
    FAISS_THROW_IF_NOT(sq_storage != nullptr);

    const float* database = x;
    std::vector<float> database_copy;
    if (old_total > 0) {
        database_copy.resize(total * index.d);
        if (index.original_database.size() ==
            static_cast<size_t>(old_total * index.d)) {
            memcpy(
                database_copy.data(),
                index.original_database.data(),
                sizeof(float) * old_total * index.d);
        } else {
            for (idx_t i = 0; i < old_total; i++) {
            index.reconstruct(i, database_copy.data() + i * index.d);
            }
        }
        memcpy(
                database_copy.data() + old_total * index.d,
                x,
                sizeof(float) * n * index.d);
        database = database_copy.data();
    }

        std::vector<float> original_database_copy(
            database, database + total * index.d);

    clear_hnswsq_block_state(index);
    sq_storage->reset();
    index.hnsw.reset();

    if (total < IndexHNSWSQ::kBlockSize ||
        total % IndexHNSWSQ::kBlockSize != 0) {
        sq_storage->add(total, database);
        index.ntotal = sq_storage->ntotal;
        return;
    }

    idx_t nblocks = total / IndexHNSWSQ::kBlockSize;

    ClusteringParameters cp;
    cp.min_points_per_centroid = 1;
    cp.max_points_per_centroid = std::max<int>(
            cp.max_points_per_centroid,
            static_cast<int>(IndexHNSWSQ::kBlockSize * 4));
    cp.spherical = index.metric_type == METRIC_INNER_PRODUCT;
    IndexFlat clustering_index(index.d, index.metric_type);
    Clustering clustering(index.d, nblocks, cp);
    clustering.train(total, database, clustering_index);

    index.block_centroids = clustering.centroids;
    rebuild_hnswsq_centroid_storage(index);
        index.original_database = std::move(original_database_copy);

        std::vector<idx_t> block_members;
        collect_block_members(
            database,
            total,
            index.d,
                index.block_centroids.data(),
                nblocks,
                index.metric_type,
            IndexHNSWSQ::kBlockSize,
            block_members);

    index.block_offsets.resize(nblocks + 1);
    index.block_offsets[0] = 0;
    for (idx_t block_id = 0; block_id < nblocks; block_id++) {
        index.block_offsets[block_id + 1] =
                index.block_offsets[block_id] + IndexHNSWSQ::kBlockSize;
    }

    index.reordered_to_original = block_members;
    index.original_to_reordered.clear();

    std::vector<float> encode_batch(IndexHNSWSQ::kBlockSize * index.d);
    for (idx_t i0 = 0; i0 < total; i0 += IndexHNSWSQ::kBlockSize) {
        idx_t batch_size = std::min<idx_t>(IndexHNSWSQ::kBlockSize, total - i0);
        for (idx_t j = 0; j < batch_size; j++) {
            idx_t original_id = index.reordered_to_original[i0 + j];
            memcpy(
                    encode_batch.data() + j * index.d,
                    database + original_id * index.d,
                    sizeof(float) * index.d);
        }
        sq_storage->add(batch_size, encode_batch.data());
    }
    index.ntotal = sq_storage->ntotal;

    const auto& config = get_hnswsq_block_search_config();
    int centroid_M = index.hnsw.nb_neighbors(1);
    index.centroid_hnsw = HNSW(centroid_M);
    index.centroid_hnsw.efConstruction =
            config.centroid_ef_construction > 0 ? config.centroid_ef_construction
                                                : index.hnsw.efConstruction;
    index.centroid_hnsw.efSearch =
            config.centroid_ef_search > 0 ? config.centroid_ef_search
                                          : index.hnsw.efSearch;

    IndexHNSW centroid_index(&index.centroid_storage, centroid_M);
    centroid_index.own_fields = false;
    centroid_index.hnsw = index.centroid_hnsw;
    centroid_index.ntotal = index.centroid_storage.ntotal;
    hnsw_add_vertices(
            centroid_index,
            0,
            nblocks,
            index.block_centroids.data(),
            index.verbose,
            false);
    index.centroid_hnsw = centroid_index.hnsw;
}

} // namespace

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

IndexHNSWSQ::IndexHNSWSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSW(new IndexScalarQuantizer(d, qtype, metric), M),
                    centroid_storage(d, ScalarQuantizer::QT_bf16, metric),
          centroid_hnsw(M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() = default;

void IndexHNSWSQ::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWSQ with an underlying SQ storage");
    FAISS_THROW_IF_NOT(is_trained);

    const auto& config = get_hnswsq_block_search_config();
    use_block_centroid_search = config.use_block_centroid_search;

    // Fall back to the standard HNSWSQ path unless this is BF16 and the
    // optional block-centroid mode is explicitly enabled.
    if (!is_hnswsq_bf16(*this) || !config.use_block_centroid_search) {
        IndexHNSW::add(n, x);
        return;
    }

    rebuild_hnswsq_bf16_blocks(*this, n, x);
}

void IndexHNSWSQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    auto& mutable_self = const_cast<IndexHNSWSQ&>(*this);
    clear_hnswsq_search_profile(mutable_self);

    const auto& config = get_hnswsq_block_search_config();

    // The custom centroid-HNSW + block-scan search path is BF16-only.
    if (!is_hnswsq_bf16(*this) || !config.use_block_centroid_search ||
        block_offsets.empty() || block_centroids.empty()) {
        storage->search(n, x, k, distances, labels, params);
        return;
    }

    const IDSelector* sel = params ? params->sel : nullptr;
    idx_t nblocks = block_offsets.size() - 1;

    int centroid_ef_search = config.centroid_ef_search;
    if (centroid_ef_search <= 0) {
        centroid_ef_search = centroid_hnsw.efSearch;
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            centroid_ef_search = hnsw_params->efSearch;
        }
    }

    const double total_t0 = getmillisecs();
    double centroid_search_time_s = 0.0;
    double centroid_distance_compute_time_s = 0.0;
    double block_scan_time_s = 0.0;
    idx_t centroid_hnsw_searches = 0;
    idx_t centroid_hnsw_candidate_exhaustions = 0;
    idx_t centroid_hnsw_distance_computations = 0;
    idx_t centroid_hnsw_hops = 0;
    idx_t blocks_scanned = 0;
    idx_t batch16_blocks_scanned = 0;
    idx_t scalar_blocks_scanned = 0;
    idx_t vectors_scanned = 0;

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> storage_dis(
                storage_distance_computer(storage));

        SearchParametersHNSW centroid_params;
        centroid_params.efSearch = centroid_ef_search;

    #pragma omp for reduction(+ : centroid_search_time_s, centroid_distance_compute_time_s, block_scan_time_s, centroid_hnsw_searches, centroid_hnsw_candidate_exhaustions, centroid_hnsw_distance_computations, centroid_hnsw_hops, blocks_scanned, batch16_blocks_scanned, scalar_blocks_scanned, vectors_scanned)
        for (idx_t qi = 0; qi < n; qi++) {
            float* query_distances = distances + qi * k;
            idx_t* query_labels = labels + qi * k;
            maxheap_heapify(k, query_distances, query_labels);

            idx_t top_b = config.top_b > 0 ? config.top_b : k;
            top_b = std::max<idx_t>(1, std::min<idx_t>(top_b, nblocks));

            std::vector<float> centroid_distances(top_b);
            std::vector<idx_t> centroid_labels(top_b);
            using RH = HeapBlockResultHandler<HNSW::C>;
            RH centroid_results(
                    1,
                    centroid_distances.data(),
                    centroid_labels.data(),
                    top_b);

            IndexHNSW centroid_index(const_cast<IndexScalarQuantizer*>(&centroid_storage),
                                     centroid_hnsw.nb_neighbors(1));
            centroid_index.own_fields = false;
            centroid_index.hnsw = centroid_hnsw;
            centroid_index.ntotal = centroid_storage.ntotal;

            const double centroid_t0 = getmillisecs();
                RH::SingleResultHandler centroid_result(centroid_results);
                VisitedTable centroid_vt(
                    centroid_index.ntotal,
                    centroid_index.hnsw.use_visited_hashset);
                auto centroid_dis = std::make_unique<ProfilingDistanceComputer>(
                    std::unique_ptr<DistanceComputer>(
                        storage_distance_computer(centroid_index.storage)));
                centroid_result.begin(0);
                centroid_dis->set_query(x + qi * d);
                HNSWStats centroid_stats = centroid_index.hnsw.search(
                    *centroid_dis,
                    &centroid_index,
                    centroid_result,
                    centroid_vt,
                    &centroid_params);
                centroid_result.end();
            centroid_search_time_s += (getmillisecs() - centroid_t0) * 1e-3;
                centroid_distance_compute_time_s +=
                    centroid_dis->distance_compute_time_s;
                centroid_hnsw_searches += static_cast<idx_t>(centroid_stats.n1);
                centroid_hnsw_candidate_exhaustions +=
                    static_cast<idx_t>(centroid_stats.n2);
                centroid_hnsw_distance_computations +=
                    static_cast<idx_t>(centroid_stats.ndis);
                centroid_hnsw_hops += static_cast<idx_t>(centroid_stats.nhops);

            storage_dis->set_query(x + qi * d);
            const double block_scan_t0 = getmillisecs();
            for (idx_t bi = 0; bi < top_b; bi++) {
                idx_t block_id = centroid_labels[bi];
                if (block_id < 0) {
                    continue;
                }
                const idx_t block_begin = block_offsets[block_id];
                const idx_t block_end = block_offsets[block_id + 1];
                const idx_t block_count = block_end - block_begin;
                blocks_scanned++;
                vectors_scanned += block_count;

                if (!sel && block_count == kBlockSize) {
                    batch16_blocks_scanned++;
                    idx_t block_indices[kBlockSize];
                    float block_distances[kBlockSize];
                    for (idx_t j = 0; j < kBlockSize; ++j) {
                        block_indices[j] = block_begin + j;
                    }
                    storage_dis->distances_batch_16(
                            block_indices,
                            static_cast<size_t>(kBlockSize),
                            block_distances);

                    for (idx_t j = 0; j < kBlockSize; ++j) {
                        if (block_distances[j] < query_distances[0]) {
                            maxheap_replace_top(
                                    k,
                                    query_distances,
                                    query_labels,
                                    block_distances[j],
                                    reordered_to_original[block_begin + j]);
                        }
                    }
                    continue;
                }

                scalar_blocks_scanned++;

                for (idx_t j = block_begin; j < block_end; j++) {
                    idx_t original_id = reordered_to_original[j];
                    if (sel && !sel->is_member(original_id)) {
                        continue;
                    }
                    float distance = (*storage_dis)(j);
                    if (distance < query_distances[0]) {
                        maxheap_replace_top(
                                k,
                                query_distances,
                                query_labels,
                                distance,
                                original_id);
                    }
                }
            }
            block_scan_time_s += (getmillisecs() - block_scan_t0) * 1e-3;

            maxheap_reorder(k, query_distances, query_labels);
            if (is_similarity_metric(metric_type)) {
                for (idx_t j = 0; j < k; j++) {
                    query_distances[j] = -query_distances[j];
                }
            }
        }
    }

            mutable_self.last_search_profile.total_time_s =
                (getmillisecs() - total_t0) * 1e-3;
            mutable_self.last_search_profile.centroid_search_time_s =
                centroid_search_time_s;
                mutable_self.last_search_profile.centroid_distance_compute_time_s =
                    centroid_distance_compute_time_s;
                mutable_self.last_search_profile.centroid_traversal_time_s = std::max(
                    0.0,
                    centroid_search_time_s - centroid_distance_compute_time_s);
            mutable_self.last_search_profile.block_scan_time_s =
                block_scan_time_s;
            mutable_self.last_search_profile.queries = n;
            mutable_self.last_search_profile.centroid_hnsw_searches =
                centroid_hnsw_searches;
            mutable_self.last_search_profile.centroid_hnsw_candidate_exhaustions =
                centroid_hnsw_candidate_exhaustions;
            mutable_self.last_search_profile.centroid_hnsw_distance_computations =
                centroid_hnsw_distance_computations;
            mutable_self.last_search_profile.centroid_hnsw_hops =
                centroid_hnsw_hops;
            mutable_self.last_search_profile.blocks_scanned = blocks_scanned;
            mutable_self.last_search_profile.batch16_blocks_scanned =
                batch16_blocks_scanned;
            mutable_self.last_search_profile.scalar_blocks_scanned =
                scalar_blocks_scanned;
            mutable_self.last_search_profile.vectors_scanned = vectors_scanned;
}

void IndexHNSWSQ::reconstruct(idx_t key, float* recons) const {
    if (!original_database.empty()) {
        FAISS_THROW_IF_NOT_FMT(
                key >= 0 && key < ntotal,
                "key %" PRId64 " out of bounds",
                int64_t(key));
        memcpy(
                recons,
                original_database.data() + key * d,
                sizeof(float) * d);
        return;
    }

    if (!original_to_reordered.empty()) {
        FAISS_THROW_IF_NOT_FMT(
                key >= 0 && key < original_to_reordered.size(),
                "key %" PRId64 " out of bounds",
                int64_t(key));
        storage->reconstruct(original_to_reordered[key], recons);
        return;
    }
    storage->reconstruct(key, recons);
}

void IndexHNSWSQ::reset() {
    clear_hnswsq_block_state(*this);
    IndexHNSW::reset();
}

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

IndexHNSW2Level::IndexHNSW2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    stats.n1++;
    if (candidates.size() == 0) {
        stats.n2++;
    }

    return nres;
}

} // namespace

void IndexHNSW2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x, k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            // visited table (not hash set) for tri-state flags.
            VisitedTable vt(ntotal, /*use_hashset=*/false);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

            constexpr int candidates_size = 1;
            MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0) {
                        break;
                    }
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();

                for (int j = 0; j < k; j++) {
                    if (idxi[j] < 0) {
                        break;
                    }
                    candidates.push(idxi[j], simi[j]);
                }

                // reorder from sorted to heap
                maxheap_heapify(k, simi, idxi, simi, idxi, k);

                HNSWStats search_stats;
                search_from_candidates_2(
                        hnsw,
                        *dis,
                        k,
                        idxi,
                        simi,
                        candidates,
                        vt,
                        search_stats,
                        0,
                        k);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                ndis += search_stats.ndis;
                nhops += search_stats.nhops;

                vt.advance();
                vt.advance();

                maxheap_reorder(k, simi, idxi);
            }
        }

        hnsw_stats.combine({n1, n2, ndis, nhops});
    }
}

void IndexHNSW2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

/**************************************************************
 * IndexHNSWCagra implementation
 **************************************************************/

IndexHNSWCagra::IndexHNSWCagra() {
    is_trained = true;
}

IndexHNSWCagra::IndexHNSWCagra(
        int d,
        int M,
        MetricType metric,
        NumericType numeric_type)
        : IndexHNSW(d, M, metric) {
    FAISS_THROW_IF_NOT_MSG(
            ((metric == METRIC_L2) || (metric == METRIC_INNER_PRODUCT)),
            "unsupported metric type for IndexHNSWCagra");
    numeric_type_ = numeric_type;
    if (numeric_type == NumericType::Float32) {
        // Use flat storage with full precision for fp32
        storage = (metric == METRIC_L2)
                ? static_cast<Index*>(new IndexFlatL2(d))
                : static_cast<Index*>(new IndexFlatIP(d));
    } else if (numeric_type == NumericType::Float16) {
        auto qtype = ScalarQuantizer::QT_fp16;
        storage = new IndexScalarQuantizer(d, qtype, metric);
    } else {
        FAISS_THROW_MSG(
                "Unsupported numeric_type: only F16 and F32 are supported for IndexHNSWCagra");
    }

    metric_arg = storage->metric_arg;

    own_fields = true;
    is_trained = true;
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexHNSWCagra::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexHNSW::add(n, x);
}

void IndexHNSWCagra::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexHNSW::search(n, x, k, distances, labels, params);
    } else {
        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(this->storage));
            dis->set_query(x + i * d);
            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                auto distance = (*dis)(idx);
                if (distance < nearest_d[i]) {
                    nearest[i] = idx;
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

        search_level_0(
                n,
                x,
                k,
                nearest.data(),
                nearest_d.data(),
                distances,
                labels,
                1, // n_probes
                1, // search_type
                params);
    }
}

faiss::NumericType IndexHNSWCagra::get_numeric_type() const {
    return numeric_type_;
}

void IndexHNSWCagra::set_numeric_type(faiss::NumericType numeric_type) {
    numeric_type_ = numeric_type;
}

} // namespace faiss
