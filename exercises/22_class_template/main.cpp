#include "../exercise.h"
#include <cstring>
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        for(int i=0;i<4; ++i){
            shape[i] = shape_[i];
            size *=shape[i];
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
    // 1) 检查：单向广播（others -> this）
    for (int d = 0; d < 4; ++d) {
        if (!(others.shape[d] == 1 || others.shape[d] == shape[d])) {
            throw std::runtime_error("operator+=: others cannot be broadcast to this shape");
        }
    }

    // 2) stride_this：this 的连续内存步长（NCHW 风格）
    unsigned int stride_this[4];
    stride_this[3] = 1;
    for (int d = 2; d >= 0; --d) {
        stride_this[d] = stride_this[d + 1] * shape[d + 1];
    }

    // 3) stride_other：others 的连续内存步长（一定要用 others.shape）
    unsigned int stride_other[4];
    stride_other[3] = 1;
    for (int d = 2; d >= 0; --d) {
        stride_other[d] = stride_other[d + 1] * others.shape[d + 1];
    }

    // 4) 遍历 this 的每个元素 idx，并找到对应的 others 元素 oidx
    unsigned int total = shape[0] * shape[1] * shape[2] * shape[3];

    for (unsigned int idx = 0; idx < total; ++idx) {
        // 4.1) idx -> 4D 坐标 coord（在 this 的坐标系里）
        unsigned int rem = idx;
        unsigned int coord[4];
        for (int d = 0; d < 4; ++d) {
            coord[d] = rem / stride_this[d];
            rem %= stride_this[d];
        }

        // 4.2) this 的 coord -> others 的线性索引 oidx（广播维度用 0）
        unsigned int oidx = 0;
        for (int d = 0; d < 4; ++d) {
            unsigned int oc = (others.shape[d] == 1) ? 0 : coord[d];
            oidx += oc * stride_other[d];
        }

        // 4.3) 执行 +=
        data[idx] += others.data[oidx];
    }

    return *this;
}

};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
