#include "../exercise.h"
#include <vector>

// 添加类型别名定义
using udim = unsigned int;

// 张量步长计算
std::vector<udim> strides(std::vector<udim> const &shape) {
    std::vector<udim> strides(shape.size());  // 创建一个与 shape 长度相同的步长向量
    udim stride = 1;  // 最后一维的步长初始化为 1

    // 使用逆向迭代器，从最后一维开始计算步长
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;   // 当前维度的步长为上一维度的步长
        stride *= shape[i];    // 步长更新：当前维度大小 * 上一维的步长
    }

    return strides;  // 返回计算得到的步长向量
}

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    ASSERT((strides({2, 3, 4}) == std::vector<udim>{12, 4, 1}), "Make this assertion pass.");
    ASSERT((strides({3, 4, 5}) == std::vector<udim>{20, 5, 1}), "Make this assertion pass.");
    ASSERT((strides({1, 3, 224, 224}) == std::vector<udim>{150528, 50176, 224, 1}), "Make this assertion pass.");
    ASSERT((strides({7, 1, 1, 1, 5}) == std::vector<udim>{5, 5, 5, 5, 1}), "Make this assertion pass.");
    return 0;
}
