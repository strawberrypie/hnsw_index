#include <cassert>
#include <ctime>
#include <random>
#include "index.hpp"

using Vector = std::vector<float>;
using Distance = hnsw::CosineSimilarity;
using Key = u_int32_t;
using Index = hnsw::Index<Key, Vector, Distance>;
using LinearIndex = std::vector<std::pair<Key, Vector>>;

const std::string EMBEDDINGS_FILENAME = "embeddings.txt";
const std::string INDICES_FILENAME = "indices.txt";


std::vector<Index::SearchResult> linear_search(
        const Vector &target,
        LinearIndex &linear_index,
        size_t n_neighbors = 3) {
    static Distance dist;

    std::sort(linear_index.begin(), linear_index.end(),
              [&target](const std::pair<Key, Vector> &left,
                        const std::pair<Key, Vector> &right) {
                  return dist(left.second, target) < dist(right.second, target);
              });

    std::vector<Index::SearchResult> result(n_neighbors);
    for (size_t i = 0; i < std::min(n_neighbors, linear_index.size()); ++i) {
        result[i] = {linear_index[i].first, dist(linear_index[i].second, target)};
    }
    return result;
}


Vector read_vector(std::istream &embeddings, size_t n_dim) {
    Vector result(n_dim);
    for (size_t i = 0; i < n_dim; ++i) {
        embeddings >> result[i];
    }
    return result;
}


int main() {
    std::ifstream embeddings(EMBEDDINGS_FILENAME);
    std::ifstream indices(INDICES_FILENAME);

    size_t n_vectors = 0, n_dim = 0;
    embeddings >> n_vectors >> n_dim;

    size_t n_indices = 0;
    indices >> n_indices;
    assert(n_indices == n_vectors);

    std::cout << "Index size: " << n_vectors << "\n" <<
              "Dimensions: " << n_dim << std::endl;

    // TODO debug
    n_vectors = n_vectors - 1;

    std::vector<Vector> vectors(n_vectors);
    std::vector<u_int32_t> keys(n_vectors);
    for (u_int32_t i = 0; i < n_vectors; ++i) {
        auto vector = read_vector(embeddings, n_dim);
        u_int32_t key;
        indices >> key;

        vectors[i] = vector;
        keys[i] = key;
    }
    std::cout << "Vectors are read!\n" << std::endl;

    auto target_vector = read_vector(embeddings, n_dim);
    uint32_t target_key;
    indices >> target_key;
    embeddings.close();
    indices.close();

    // Initialize timings
    std::clock_t start;
    double duration;

    start = std::clock();
    LinearIndex linear_index;
    for (u_int32_t i = 0; i < n_vectors; ++i) {
        linear_index.emplace_back(std::pair(keys[i], vectors[i]));
    }
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    std::cout << "Linear index created in " << duration << " seconds" << std::endl;

    start = std::clock();
    auto index = hnsw::Index<Key, Vector, Distance>();
    for (u_int32_t i = 0; i < n_vectors; ++i) {
        index.insert(keys[i], vectors[i]);
    }
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    std::cout << "HNSW index created in " << duration << " seconds" << std::endl;
    std::cout << std::endl;

    std::cout << "HNSW index results:" << std::endl;
    std::cout << "True key:" << target_key << std::endl;

    start = std::clock();
    auto query = index.search(target_vector, 5);
    for (const auto &result : query) {
        std::cout << result.key << " " << result.distance << std::endl;
    }
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    std::cout << "Took " << duration << " seconds" << std::endl;
    std::cout << std::endl;

    std::cout << "Linear index results:" << std::endl;
    start = std::clock();
    auto linear_query = linear_search(target_vector, linear_index, 5);
    for (const auto &result : linear_query) {
        std::cout << result.key << " " << result.distance << std::endl;
    }
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    std::cout << "Took " << duration << " seconds" << std::endl;

    return 0;
}
