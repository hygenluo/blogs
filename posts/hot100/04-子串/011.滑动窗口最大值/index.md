---
title: 滑动窗口最大值
author: Hygen
description: Leetcode-239
category: hot100
tags: [Solution, 单调队列]
published: 2026-06-28
---
# 题目描述
[题目链接](https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到最右侧。你只可以看到在滑动窗口内的 `k` 个数字，滑动窗口每次只向右移动一位。

返回**滑动窗口中的最大值**组成的数组。

## 输入
整数数组 `nums`，整数 `k`
## 输出
长度为 `n - k + 1` 的整数数组，每个元素为对应窗口内的最大值
## 约束条件
1. `1 <= nums.length <= 10^5`
2. `-10^4 <= nums[i] <= 10^4`
3. `1 <= k <= nums.length`
# 进阶约束
1. 能否将时间复杂度优化到 $O(n)$？
## 示例1
```
input: nums = [1,3,-1,-3,5,3,6,7], k = 3
output: [3,3,5,5,6,7]
```
解释：窗口位置与最大值如下：

| 窗口位置 | 窗口内元素 | 最大值 |
| --- | --- | --- |
| `[1  3  -1]` | `1, 3, -1` | `3` |
| `[3  -1  -3]` | `3, -1, -3` | `3` |
| `[-1  -3  5]` | `-1, -3, 5` | `5` |
| `[-3  5  3]` | `-3, 5, 3` | `5` |
| `[5  3  6]` | `5, 3, 6` | `6` |
| `[3  6  7]` | `3, 6, 7` | `7` |

## 示例2
```
input: nums = [1], k = 1
output: [1]
```
# 解题思路
题目要求对每个长度为 `k` 的连续子数组求最大值。暴力枚举每个窗口再逐个比较，在 $n$ 可达 $10^5$、$k$ 也可能很大时会超时。能否在窗口右移时**复用**已算过的信息，而不是每次都重新扫一遍？

## 从暴力枚举出发
最直观的做法：枚举每个窗口起点 `i`，在 `nums[i:i+k]` 内找最大值：

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        n = len(nums)
        for i in range(n - k + 1):
            res.append(max(nums[i:i + k]))
        return res
```

每个窗口调用 `max` 扫 $k$ 个元素，时间复杂度 $O(nk)$。当 $k$ 接近 $n$ 时接近 $O(n^2)$，会超时。瓶颈在于：**相邻两个窗口有大量重叠元素**，却每次都从零开始比较——能否记住「谁还有资格当最大值」？

## 堆：动态维护窗口内的最大值
用**大根堆**存窗口内的 `(值, 下标)`，每次右移：

1. 把新元素 `(nums[i], i)` 入堆
2. 堆顶若已滑出窗口左边界（`下标 <= i - k`），则弹出
3. 堆顶即为当前窗口最大值

```python
import heapq

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        heap = []
        res = []
        for i, x in enumerate(nums):
            heapq.heappush(heap, (-x, i))
            if i >= k - 1:
                while heap[0][1] <= i - k:
                    heapq.heappop(heap)
                res.append(-heap[0][0])
        return res
```

堆中最多 $k$ 个元素，每次入堆、出堆 $O(\log k)$，总时间 $O(n \log k)$，比暴力优，但每个元素仍可能被多次入堆（旧的大值暂时留在堆里需惰性删除），常数也不小。能否让每个下标**最多进出数据结构一次**？

## 单调队列：只保留「还有希望成为最大值」的下标
维护一个队列 `q`，存的是**下标**（不是值本身），并保证对应数值**从队首到队尾单调递减**——队首 `q[0]` 永远是当前窗口最大值的下标。

对每个新元素 `nums[i]`：

1. **从队尾淘汰无用下标**：若 `nums[q[-1]] <= nums[i]`，则 `q[-1]` 对应的元素比 `i` 更旧且更小，在之后任何包含 `i` 的窗口里都不可能成为最大值，持续 `pop` 直到队尾更大或队空。
2. **入队**：将 `i` 追加到队尾。
3. **淘汰滑出窗口的队首**：若 `i - q[0] >= k`，说明 `q[0]` 已不在窗口 `[i-k+1, i]` 内，`pop` 队首。
4. **记录答案**：当 `i >= k - 1` 时，窗口成形，答案为 `nums[q[0]]`。

以 `nums = [1, 3, -1, -3, 5, 3, 6, 7]`、`k = 3` 为例：

| `i` | `x` | 队尾淘汰 | `q`（下标） | 对应值 | 滑出淘汰 | `res` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | — | `[0]` | `[1]` | — | — |
| 1 | 3 | 弹出 `0`（`1≤3`） | `[1]` | `[3]` | — | — |
| 2 | -1 | — | `[1,2]` | `[3,-1]` | — | `[3]` |
| 3 | -3 | — | `[1,2,3]` | `[3,-1,-3]` | — | `[3,3]` |
| 4 | 5 | 弹出 `3,2,1` | `[4]` | `[5]` | — | `[3,3,5]` |
| 5 | 3 | — | `[4,5]` | `[5,3]` | — | `[3,3,5,5]` |
| 6 | 6 | 弹出 `5,4` | `[6]` | `[6]` | — | `[3,3,5,5,6]` |
| 7 | 7 | 弹出 `6` | `[7]` | `[7]` | — | `[3,3,5,5,6,7]` |

关键观察：

- 每个下标**最多入队一次、出队一次**（队尾淘汰或队首滑出），摊还 $O(n)$。
- 队首始终是当前窗口最大值，查询为 $O(1)$。

## 复杂度
- 时间复杂度：$O(n)$，每个下标最多进出队列各一次
- 空间复杂度：$O(k)$，队列最多存 $k$ 个下标

## 代码
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = []
        res = []
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x: q.pop()
            q.append(i)
            if i - q[0] >= k: q.pop(0)
            if i >= k - 1: res.append(nums[q[0]])
        return res
```

# 解题思路（堆 vs 单调队列）
上面已给出两种非暴力思路，对比如下：

| 对比项 | 大根堆 + 惰性删除 | 单调递减队列 |
| --- | --- | --- |
| 时间 | $O(n \log k)$ | $O(n)$ |
| 每个下标进出次数 | 可能多次入堆 | 最多各一次 |
| 思路难度 | 较低，堆是常见工具 | 需理解「单调性」与「淘汰」 |
| 适用 | $k$ 很小时足够快 | 本题最优解 |

若面试时一时想不出单调队列，**堆**是可靠的 $O(n \log k)$ 备选；若要求 $O(n)$ 或 $k$ 很大，单调队列是标准正解。两者都利用了「窗口右移时淘汰过期信息」，但单调队列通过**按值单调**直接扔掉永远不可能胜出的候选，避免了堆的重复存储与惰性删除。

# 代码优化
算法层面单调队列已达 $O(n)$ 最优（至少需读一遍数组）。以下优化侧重**实现细节**：列表 `pop(0)` 会整体前移元素，单次 $O(k)$，最坏总复杂度退化为 $O(nk)$；改用双端队列可使队首删除变为 $O(1)$。

## 用 `deque` 替代列表作队列
`collections.deque` 的 `popleft()` 为 $O(1)$，语义与 `pop(0)` 相同，整体保证 $O(n)$：

```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()
        res = []
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            if i - q[0] >= k:
                q.popleft()
            if i >= k - 1:
                res.append(nums[q[0]])
        return res
```

## 合并条件，减少嵌套
窗口滑出判断可与答案记录写在同一逻辑块，用早期 `continue` 跳过未成形的窗口，主流程更清晰：

```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()
        res = []
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            if i < k - 1:
                continue
            if q[0] <= i - k:
                q.popleft()
            res.append(nums[q[0]])
        return res
```

注意此处先判断 `i >= k - 1` 再淘汰队首，与原版「先淘汰再取答案」等价：当 `i = k - 1` 时，队首下标最小为 `0`，`0 <= i - k` 不成立，不会误删。两种写法时间均为 $O(n)$，推荐 `deque` + 适当换行以兼顾性能与可读性。
