#include <cmath>
#include <omp.h>

namespace hnsw {

namespace impl {

template<typename T>
T l2_square_distance(const T *left, const T *right, size_t dim) {
    T sum = 0;

    #pragma omp simd
    for (size_t i = 0; i < dim; ++i) {
        T diff = left[i] - right[i];
        sum += diff * diff;
    }

    return sum;
}

template<typename T>
T cosine_similarity(const T *left, const T *right, size_t dim) {
    T sum = 0;
    T sum_left = 0;
    T sum_right = 0;

    #pragma omp simd
    for (size_t i = 0; i < dim; ++i) {
        sum += left[i] * right[i];
        sum_left += left[i] * left[i];
        sum_right += right[i] * right[i];
    }

    if (sum_left < 2 * std::numeric_limits<T>::min()) {
        if (sum_right < 2 * std::numeric_limits<T>::min()) {
            return T(1.0);
        } else {
            return T(0.0);
        }
    }

    T cosine_distance = sum / (std::sqrt(sum_left) * std::sqrt(sum_right));

    return T(1.0) - std::max(T(-1.0), std::min(T(1.0), cosine_distance));
}

}

struct L2SquareDistance {

    template<typename Vector>
    auto operator()(const Vector &one, const Vector &another) const {
        if (one.size() != another.size()) {
            throw std::runtime_error("L2SquareDistance: vectors sizes do not match");
        }

        return impl::l2_square_distance(one.data(), another.data(), one.size());
    }
};

struct CosineSimilarity {

    template<typename Vector>
    auto operator()(const Vector &one, const Vector &another) const {
        if (one.size() != another.size()) {
            throw std::runtime_error("CosineDistance: vectors sizes do not match");
        }

        return impl::cosine_similarity(one.data(), another.data(), one.size());
    }
};

}