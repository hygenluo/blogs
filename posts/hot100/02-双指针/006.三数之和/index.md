---
title: 三数之和
author: Hygen
description: Leetcode-15
category: hot100
tags: [Solution, 双指针]
published: 2026-06-17
---
# 题目描述
[题目链接](https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-100-liked)

给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k`，同时还满足 `nums[i] + nums[j] + nums[k] == 0`。请你返回所有和为 `0` 且不重复的三元组。

**注意：** 答案中不可以包含重复的三元组。
## 输入
整数数组 `nums`
## 输出
所有和为 `0` 且不重复的三元组列表
## 约束条件
1. `3 <= nums.length <= 3000`
2. `-10^5 <= nums[i] <= 10^5`
# 进阶约束
1. 输出中不得包含重复的三元组
## 示例1
```
input: nums = [-1,0,1,2,-1,-4]
output: [[-1,-1,2],[-1,0,1]]
```
解释：和为 `0` 的三元组有 `[-1, 0, 1]` 与 `[-1, -1, 2]`，输出顺序不限。
## 示例2
```
input: nums = [0,1,1]
output: []
```
## 示例3
```
input: nums = [0,0,0]
output: [[0,0,0]]
```
# 解题思路
题目本质是：在数组中找出所有**不重复**的三元组，使其元素之和为 `0`。难点不在于「找不找得到」，而在于**去重**——同一个三元组可能对应多组下标，答案里只能出现一次。

## 从暴力枚举出发
最直观的做法是三层循环，枚举所有 `(i, j, k)` 组合，检查三数之和是否为 `0`：

```python
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        res = []
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if nums[i] + nums[j] + nums[k] == 0:
                        res.append(sorted([nums[i], nums[j], nums[k]]))
        return list({tuple(t) for t in res})  # 借助 set 去重
```

思路正确，但时间复杂度为 $O(n^3)$，在 $n$ 可达 $3000$ 时会超时；末尾还要额外去重，写法也繁琐。我们需要减少枚举量，并更优雅地处理重复。

## 降维：固定一个数，转化为两数之和
若固定第一个数 `nums[i]`，问题就退化为：在剩余元素中找两个数，使三数之和为 `0`，即找 `nums[j] + nums[k] = -nums[i]`——这正是经典的**两数之和**。

两数之和可以用哈希表 $O(n)$ 求解，但本题要返回**全部**不重复三元组，哈希法去重较麻烦。更自然的做法是：**先排序，再用双指针**在有序数组上 $O(n)$ 找一对数。

## 先排序
对 `nums` 排序后有两个好处：

1. **双指针可行**：有序数组上，左右指针向内移动时，三数之和单调变化，可以决定移动哪一侧
2. **去重简单**：相同元素相邻排列，跳过重复值即可

```python
nums.sort()
```

## 外层枚举第一个数
用下标 `i` 枚举三元组的第一个元素，范围是 `[0, n - 3]`（后面至少还要留两个位置给 `j` 和 `k`）。

### 跳过重复的 `nums[i]`
若 `nums[i]` 与 `nums[i - 1]` 相同，以它为第一个数得到的三元组必然重复，直接 `continue`：

```python
if i and nums[i] == nums[i - 1]:
    continue
```

### 剪枝：提前结束或跳过
数组已有序，可以利用边界值快速判断当前 `i` 是否还有希望：

- `nums[i] + nums[i+1] + nums[i+2] > 0`：最小的三个数之和已大于 `0`，后面 `i` 更大只会更糟，**直接 `break`**
- `nums[i] + nums[-1] + nums[-2] < 0`：即使配上最大的两个数仍小于 `0`，当前 `i` 不可能有解，**`continue` 换下一个 `i`**

## 内层双指针：在 `[i+1, n-1]` 上找两数之和
设 `j = i + 1`、`k = n - 1`，在 `j < k` 时计算 `cnt = nums[i] + nums[j] + nums[k]`：

| 情况 | 操作 |
| --- | --- |
| `cnt < 0` | 和太小，`j` 右移，尝试更大的 `nums[j]` |
| `cnt > 0` | 和太大，`k` 左移，尝试更小的 `nums[k]` |
| `cnt == 0` | 找到一组解，记录后 `j`、`k` 同时向内移动 |

以 `nums = [-4, -1, -1, 0, 1, 2]`（排序后）为例，当 `i = 1`（`nums[i] = -1`）时：

| `j` | `k` | `cnt` | 操作 |
| --- | --- | --- | --- |
| 2 | 5 | -1 + (-1) + 2 = 0 | 记录 `[-1, -1, 2]`，`j`、`k` 内移并去重 |
| 3 | 4 | -1 + 0 + 1 = 0 | 记录 `[-1, 0, 1]`，`j`、`k` 内移 |

### 找到解后跳过重复的 `j`、`k`
命中一组解后，`j` 和 `k` 各自跳过与当前值相同的元素，避免重复三元组：

```python
while j < k and nums[j] == nums[j - 1]:
    j += 1
while j < k and nums[k] == nums[k + 1]:
    k -= 1
```

## 复杂度
- 时间复杂度：$O(n^2)$。排序 $O(n \log n)$，外层 `i` 遍历 $O(n)$，内层双指针总共移动 $O(n)$ 次
- 空间复杂度：$O(1)$（不计排序栈空间和输出数组）

## 代码
```python
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res = []
        n = len(nums)
        for i in range(n - 2):
            if i and nums[i] == nums[i - 1]: continue
            if nums[i] + nums[i + 1] + nums[i + 2] > 0: break
            if nums[i] + nums[-1] + nums[-2] < 0: continue
            j, k = i + 1, n - 1
            while j < k:
                cnt = nums[i] + nums[j] + nums[k]
                if cnt < 0: j += 1
                elif cnt > 0: k -= 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    while j < k and nums[j] == nums[j - 1]: j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]: k -= 1

        return res
```

# 解题思路（哈希表）
排序 + 双指针已是本题的主流最优解，时间 $O(n^2)$、空间 $O(1)$。另一种常见思路是**不排序，用哈希表做两数之和**：

1. 外层仍枚举第一个数 `nums[i]`
2. 内层用哈希集合 `seen` 记录已遍历过的 `nums[j]`
3. 对每个 `j`，目标值为 `target = -nums[i] - nums[j]`；若 `target in seen`，说明之前某处出现过补数，可组成三元组
4. 将 `nums[j]` 加入 `seen`

去重策略：对外层 `i` 跳过重复值；命中解后将三元组排序后存入 `set`，最后转回列表。

```python
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        res = set()
        n = len(nums)
        for i in range(n - 2):
            if i and nums[i] == nums[i - 1]:
                continue
            seen = set()
            for j in range(i + 1, n):
                complement = -nums[i] - nums[j]
                if complement in seen:
                    res.add(tuple(sorted((nums[i], complement, nums[j]))))
                seen.add(nums[j])
        return [list(t) for t in res]
```

| 对比项 | 排序 + 双指针 | 哈希表 |
| --- | --- | --- |
| 时间 | $O(n^2)$ | $O(n^2)$ |
| 空间 | $O(1)$ | $O(n)$ |
| 去重 | 利用有序性，逻辑清晰 | 需借助 `set` + 排序元组 |
| 剪枝 | 边界剪枝效果好 | 较难做同类剪枝 |

面试和竞赛中更推荐排序 + 双指针：空间更优、去重更自然，也便于扩展到「三数之和最接近」「四数之和」等变形题。

# 代码优化
本题在算法层面，$O(n^2)$ 已接近最优——最坏情况下合法三元组数量可达 $O(n^2)$，输出本身就需要 $O(n^2)$ 时间。因此优化空间主要在**剪枝**与**代码可读性**。

## 缓存 `nums[i]`，减少重复下标访问
外层循环中多次用到 `nums[i]`，可先存入局部变量：

```python
for i in range(n - 2):
    x = nums[i]
    if i and x == nums[i - 1]:
        continue
    if x + nums[i + 1] + nums[i + 2] > 0:
        break
    if x + nums[-1] + nums[-2] < 0:
        continue
    # 内层用 x 代替 nums[i]
```

## 抽取「跳过重复值」为内联模式
找到解后 `j`、`k` 的去重逻辑对称，可先各自前进一步再统一跳过：

```python
else:
    res.append([nums[i], nums[j], nums[k]])
    j += 1
    k -= 1
    while j < k and nums[j] == nums[j - 1]:
        j += 1
    while j < k and nums[k] == nums[k + 1]:
        k -= 1
```

这与原写法等价，但将「记录解」与「移动指针」和「去重」三步分开，阅读时层次更清楚。

## 命中解时双指针同时内移
`cnt == 0` 时 `j += 1` 与 `k -= 1` 必须**都执行**——无论 `nums[j]` 与 `nums[k]` 是否相等，当前这对 `(j, k)` 已被消费，只移动一侧会重复枚举同一对。原代码在 `j += 1` 后做 `j` 去重、再 `k -= 1` 后做 `k` 去重，顺序正确，是常见写法。
