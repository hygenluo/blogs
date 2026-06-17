---
title: 盛水最多的容器
author: Hygen
description: Leetcode-11
category: hot100
tags: [Solution, 双指针]
published: 2026-06-17
---
# 题目描述
[题目链接](https://leetcode.cn/problems/container-with-most-water/?envType=study-plan-v2&envId=top-100-liked)

给定一个长度为 `n` 的整数数组 `height`。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])`。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：** 你不能倾斜容器。
## 输入
整数数组 `height`
## 输出
容器可以储存的最大水量（整数）
## 约束条件
1. `n == height.length`
2. `2 <= n <= 10^5`
3. `0 <= height[i] <= 10^4`
# 进阶约束
1. 能否将时间复杂度优化到 $O(n)$？
## 示例1
```
input: height = [1,8,6,2,5,4,8,3,7]
output: 49
```
解释：选择下标 `1` 和 `8` 的两条线，高度为 `8` 和 `7`，宽度为 `7`，面积为 `min(8, 7) × 7 = 49`。
## 示例2
```
input: height = [1,1]
output: 1
```
# 解题思路
题目本质是：在数组中选两个下标 `i < j`，以它们为左右边界、以 `min(height[i], height[j])` 为「短板」高度，计算容器面积 `min(height[i], height[j]) × (j - i)`，求最大值。

## 从暴力枚举出发
最直观的做法是枚举所有 `(i, j)` 配对，逐一计算面积并取最大值：

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        n = len(height)
        for i in range(n):
            for j in range(i + 1, n):
                ans = max(ans, min(height[i], height[j]) * (j - i))
        return ans
```

思路正确，但时间复杂度为 $O(n^2)$，在 $n$ 可达 $10^5$ 时会超时。我们需要找到一种能**跳过大量无效配对**的方法。

## 观察：宽度与高度的取舍
面积由两个因素决定：

- **宽度** `(j - i)`：两指针越靠近，宽度越小
- **高度** `min(height[i], height[j])`：由较短的那条边决定（木桶效应）

若固定左右边界后向内收缩，宽度必然减小。此时若想面积变大，只能指望高度增加——但高度受短板限制，**移动较高的一边**只会让宽度变小，而高度仍被原来的短板卡住，面积一定变小，可以直接排除。

反过来，**移动较短的一边**虽然宽度也减小了，但有机会遇到更高的线，从而抬高短板，面积仍有可能变大。

## 双指针：从两端向中间收缩
基于上述观察，用两个指针 `l` 和 `r` 分别指向数组首尾，初始时宽度最大：

1. 计算当前面积 `min(height[l], height[r]) × (r - l)`，更新答案
2. 比较 `height[l]` 与 `height[r]`，将**较短一侧**的指针向内移动一步
3. 重复直到 `l >= r`

以 `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]` 为例，前几步如下：

| `l` | `r` | `height[l]` | `height[r]` | 面积 | 移动 |
| --- | --- | --- | --- | --- | --- |
| 0 | 8 | 1 | 7 | 1 × 8 = 8 | `l` 右移（左侧更短） |
| 1 | 8 | 8 | 7 | 7 × 7 = **49** | `r` 左移（右侧更短） |
| 1 | 7 | 8 | 3 | 3 × 6 = 18 | `r` 左移 |
| 1 | 6 | 8 | 8 | 8 × 5 = 40 | 相等，移动 `l` 或 `r` 均可 |

遍历过程中记录到的最大面积即为 `49`。

为什么这样不会漏掉最优解？假设最优配对为 `(i, j)`，当指针尚未到达它们时，若较短边在外侧，双指针会不断将其向内推；一旦某一侧指针越过 `i` 或 `j`，说明所有以该外侧边为短板的更宽配对都已检查完毕。每个指针最多移动 $n - 1$ 次，因此总共 $O(n)$ 步。

## 复杂度
- 时间复杂度：$O(n)$，两个指针各最多移动 $n - 1$ 次
- 空间复杂度：$O(1)$，仅使用常数额外空间

## 代码
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        l, r = 0, len(height) - 1
        while l < r:
            ans = max(ans, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]: l += 1
            else: r -= 1

        return ans
```

# 代码优化
本题在算法层面，双指针 $O(n)$ 已是时间最优——不存在比线性更低的下界（至少要看一遍数组）。因此优化空间主要在代码层面。

## 合并相等时的分支
当 `height[l] == height[r]` 时，移动左指针或右指针效果等价：宽度都减 1，且两侧高度相同，新面积不可能超过当前。将条件改为 `<=` 可统一写法，避免 `if-else` 两路分支：

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        l, r = 0, len(height) - 1
        while l < r:
            ans = max(ans, min(height[l], height[r]) * (r - l))
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return ans
```

## 减少重复下标访问
在循环较密集时，可将两侧高度先存入局部变量，避免多次通过下标读取数组（对 Python 而言提升有限，但意图更清晰）：

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        l, r = 0, len(height) - 1
        while l < r:
            hl, hr = height[l], height[r]
            ans = max(ans, min(hl, hr) * (r - l))
            if hl <= hr:
                l += 1
            else:
                r -= 1
        return ans
```
