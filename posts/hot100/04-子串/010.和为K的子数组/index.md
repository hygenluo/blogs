---
title: 和为 K 的子数组
author: Hygen
description: Leetcode-560
category: hot100
tags: [Solution, 前缀和]
published: 2026-06-28
---
# 题目描述
[题目链接](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked)

给定一个整数数组 `nums` 和一个整数 `k`，请你统计并返回该数组中**和为 `k` 的连续子数组**的个数。

## 输入
整数数组 `nums`，整数 `k`
## 输出
和为 `k` 的连续子数组个数
## 约束条件
1. `1 <= nums.length <= 2 * 10^4`
2. `-1000 <= nums[i] <= 1000`
3. `-10^7 <= k <= 10^7`
# 进阶约束
1. 能否将时间复杂度优化到 $O(n)$？
## 示例1
```
input: nums = [1,1,1], k = 2
output: 2
```
解释：和为 `2` 的子数组有 `[1,1]`（下标 `0~1`）与 `[1,1]`（下标 `1~2`），共 `2` 个。
## 示例2
```
input: nums = [1,2,3], k = 3
output: 2
```
解释：和为 `3` 的子数组有 `[1,2]`（下标 `0~1`）与 `[3]`（下标 `2`），共 `2` 个。
# 解题思路
题目要求统计**连续**子数组中和恰好等于 `k` 的个数。暴力枚举所有子区间再逐个求和，在 $n$ 可达 $2 \times 10^4$ 时会超时。能否把「区间求和」转化为某种**可复用的前缀信息**，并在一次扫描中完成统计？

## 从暴力枚举出发
最直观的做法：枚举每个起点 `i` 和终点 `j`，计算 `nums[i:j+1]` 的和，等于 `k` 则计数：

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        n = len(nums)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += nums[j]
                if s == k:
                    ans += 1
        return ans
```

内层循环从 `i` 向右累加，避免重复计算区间和，时间复杂度 $O(n^2)$，仍会超时。瓶颈在于：对每个终点 `j`，我们要知道**有多少个起点** `i`，使得 `nums[i:j+1]` 的和为 `k`——能否不逐个枚举 `i`？

## 前缀和：把区间和变成两个前缀的差
设 `prefix[j]` 为 `nums[0..j]` 的前缀和，则子数组 `nums[i+1..j]` 的和为：

$$
\text{prefix}[j] - \text{prefix}[i] = k \quad \Leftrightarrow \quad \text{prefix}[i] = \text{prefix}[j] - k
$$

对每个终点 `j`，问题转化为：在 `j` 之前，有多少个前缀和恰好等于 `prefix[j] - k`？若用哈希表记录「每个前缀和出现了几次」，查询就是 $O(1)$ 的。

## 哈希表统计前缀和频次
从左到右扫描，维护当前前缀和 `Sum`，并用字典 `d` 记录已出现过的前缀和及其频次：

1. **`d[0] = 1`**：空前缀的和为 `0`，出现 `1` 次。这样当某个子数组本身（从下标 `0` 开始）的和恰好为 `k` 时，`Sum - k = 0` 也能被正确计数。
2. **加入 `nums[i]`**：`Sum += x`，当前前缀和更新。
3. **累加答案**：`ans += d[Sum - k]`，即统计此前有多少个前缀和等于 `Sum - k`，每个都对应一段和为 `k` 的子数组。
4. **更新频次**：`d[Sum] += 1`，把当前前缀和记入哈希表，供后续位置查询。

以 `nums = [1, 2, 3]`、`k = 3` 为例：

| `i` | `x` | `Sum` | `Sum - k` | `d[Sum-k]` | `ans` | `d`（更新后） |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | -2 | 0 | 0 | `{0:1, 1:1}` |
| 1 | 2 | 3 | 0 | 1 | 1 | `{0:1, 1:1, 3:1}` |
| 2 | 3 | 6 | 3 | 1 | 2 | `{0:1, 1:1, 3:1, 6:1}` |

- `i = 1` 时：`Sum = 3`，`d[0] = 1` 表示存在空前缀，对应子数组 `[1, 2]`（下标 `0~1`）。
- `i = 2` 时：`Sum = 6`，`d[3] = 1` 表示此前前缀和 `3` 出现一次（在 `i = 1`），对应子数组 `[3]`（下标 `2`）。

最终答案为 `2`。

以 `nums = [1, 1, 1]`、`k = 2` 为例：

| `i` | `x` | `Sum` | `Sum - k` | `d[Sum-k]` | `ans` |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | -1 | 0 | 0 |
| 1 | 1 | 2 | 0 | 1 | 1 |
| 2 | 1 | 3 | 1 | 1 | 2 |

`i = 1` 时命中空前缀，得到 `[1, 1]`（下标 `0~1`）；`i = 2` 时命中前缀和 `1`，得到 `[1, 1]`（下标 `1~2`）。注意**相同数值的子数组因起点不同要分别计数**，哈希表用「频次」而非「是否出现」正是为此服务。

## 为何不用滑动窗口？
若数组元素**全为正数**，可用滑动窗口：右扩累加和，和超过 `k` 则左缩，和等于 `k` 时计数。但本题 `nums[i]` 可为负数——左缩一位后区间和可能变小，也可能变大，无法保证单调性，滑动窗口会漏解或重复计数。前缀和 + 哈希表不依赖单调性，是此题的通用正解。

## 复杂度
- 时间复杂度：$O(n)$，每个元素入表、查表各一次
- 空间复杂度：$O(n)$，哈希表最多存 $n$ 个不同的前缀和

## 代码
```python
from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        d = defaultdict(int)
        d[0] = 1
        Sum = ans = 0
        for i, x in enumerate(nums):
            Sum += x
            ans += d[Sum - k]
            d[Sum] += 1
        return ans
```

# 代码优化
算法层面前缀和 + 哈希表已达 $O(n)$ 最优（至少需读一遍数组），无法再降时间复杂度。以下优化侧重**代码规范**与**可读性**。

## 变量命名与 `dict.get`
`Sum` 按 PEP 8 宜改为 `prefix_sum` 或 `s`；若不想引入 `defaultdict`，普通字典配合 `.get(key, 0)` 语义等价，依赖更少：

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = {0: 1}
        prefix_sum = ans = 0
        for x in nums:
            prefix_sum += x
            ans += cnt.get(prefix_sum - k, 0)
            cnt[prefix_sum] = cnt.get(prefix_sum, 0) + 1
        return ans
```

## 用 `Counter` 表达「频次统计」
若更习惯「计数」语义，可将 `d` 换为 `Counter`，逻辑不变：

```python
from collections import Counter

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = Counter({0: 1})
        prefix_sum = ans = 0
        for x in nums:
            prefix_sum += x
            ans += cnt[prefix_sum - k]
            cnt[prefix_sum] += 1
        return ans
```

`Counter` 对缺失键默认返回 `0`，与 `defaultdict(int)` 行为一致。三种写法时间、空间均为 $O(n)$，按团队风格择一即可。
