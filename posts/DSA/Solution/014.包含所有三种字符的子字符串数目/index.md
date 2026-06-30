---
title: 包含所有三种字符的子字符串数目
author: Hygen
description: LeetCode-1358
category: Algorithm and Structure
tags: [Solution, 滑动窗口]
published: 2026-06-30
---
# 题目描述
[题目链接](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/description/?envType=daily-question&envId=2026-06-30)

给你一个字符串 `s`，它**只包含**三种字符 `'a'`、`'b'` 和 `'c'`。

请你返回 `s` 中**至少包含** `'a'`、`'b'` 和 `'c'` 各至少一个的子字符串数目。

## 输入
只包含字符 `'a'`、`'b'`、`'c'` 的字符串 `s`
## 输出
满足条件的子字符串数目（整数）
## 约束条件
1. `3 <= s.length <= 5 * 10^4`
2. `s` 仅由 `'a'`、`'b'` 和 `'c'` 组成
## 示例1
```
input: s = "abcabc"
output: 10
```
解释：满足条件的子字符串有：`"abc"`, `"abca"`, `"abcab"`, `"abcabc"`, `"bca"`, `"bcab"`, `"bcabc"`, `"cab"`, `"cabc"` 以及再次出现的 `"abc"`（起始位置不同）。
## 示例2
```
input: s = "aaacb"
output: 3
```
解释：满足条件的子字符串有：`"aaacb"`、`"aacb"` 和 `"acb"`。
## 示例3
```
input: s = "abc"
output: 1
```
# 解题思路
题目要求统计**至少包含** `'a'`、`'b'`、`'c'` 各一个的子字符串数量。字符串仅由这三种字符构成，这意味着一旦某个子串"集齐"了三种字符，继续向右扩展永远不会失去它们——字符只会增多，不会消失。这个单调性是后续优化的关键。

## 从暴力枚举出发
最直接的想法：枚举所有可能的起点 `i` 和终点 `j`，逐一检查子串 `s[i:j+1]` 是否包含全部三种字符，若包含则计数器加一：

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        for i in range(n):
            cnt = {'a': 0, 'b': 0, 'c': 0}
            for j in range(i, n):
                cnt[s[j]] += 1
                if cnt['a'] and cnt['b'] and cnt['c']:
                    ans += 1
        return ans
```

这种做法时间复杂度 $O(n^2)$，在 $n$ 可达 $5 \times 10^4$ 时会超时。但暴力解揭示了一个重要性质：**对固定的起点 `i`，一旦在某个位置 `j` 首次集齐三种字符，那么 `j` 之后的所有位置也都集齐了**。

## 利用单调性：一次计数多个子串
对于固定的左边界 `j`，设第一个使 `[j, i]` 满足条件的最小右边界为 `i`。由于字符串只含 `'a'`、`'b'`、`'c'`，窗口内的字符计数**只增不减**（或者说，向右扩展时已经拥有的字符不会丢失），因此对于任意 `k >= i`，子串 `[j, k]` 也一定满足条件。

也就是说，只要找到了最小的 `i`，右边界可以是 `i, i+1, ..., n-1`，一共有 $n - i$ 个合法子串。我们不必逐个枚举它们，直接累加即可。

剩下的问题是：如何高效地找到每个 `j` 对应的最小 `i`？

## 滑动窗口：左右指针各走一遍
用两个指针维护一个窗口 `[j, i]`：

- `i`：右边界，逐字符向右扩展，将 `s[i]` 加入计数
- `j`：左边界，当窗口内三种字符的计数都 $\ge 1$ 时，说明找到了最小 `i`（对当前 `j` 而言）

当窗口满足条件时，做两件事：
1. **计数**：将 $n - i$ 累加到答案（当前 `j` 配上 `i` 到 $n-1$ 的任意右边界都合法）
2. **收缩**：移除 `s[j]`，`j += 1`，尝试用更短的窗口看看是否仍满足条件——这相当于为下一个 `j` 寻找它的最小 `i`

```python
from collections import defaultdict

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ans = j = 0
        d = defaultdict(int)

        for i, c in enumerate(s):
            d[c] += 1
            while d['a'] and d['b'] and d['c']:
                ans += n - i
                d[s[j]] -= 1
                j += 1

        return ans
```

以 `s = "abcabc"` 为例跟踪执行过程：

| `i` | `c` | 窗口内计数 `{a,b,c}` | 满足条件？ | 操作 | `ans` 变化 |
|:--:|:--:|:--|:--:|------|:--:|
| 0 | `a` | {1,0,0} | ❌ | — | 0 |
| 1 | `b` | {1,1,0} | ❌ | — | 0 |
| 2 | `c` | {1,1,1} | ✅ | `ans += 6-2=4`, 移除 `s[0]=a`, `j=1` | 4 |
|   |   | {0,1,1} | ❌ | 退出 while | 4 |
| 3 | `a` | {1,1,1} | ✅ | `ans += 6-3=3`, 移除 `s[1]=b`, `j=2` | 7 |
|   |   | {1,0,1} | ❌ | 退出 while | 7 |
| 4 | `b` | {1,1,1} | ✅ | `ans += 6-4=2`, 移除 `s[2]=c`, `j=3` | 9 |
|   |   | {0,1,1} | ❌ | 退出 while | 9 |
| 5 | `c` | {1,1,1} | ✅ | `ans += 6-5=1`, 移除 `s[3]=a`, `j=4` | 10 |
|   |   | {0,1,1} | ❌ | 退出 while | 10 |

最终答案为 `10`，与预期一致。

表中可以观察到：`i` 从未回退，`j` 也只增不减——每个字符最多被 `i` 加入一次、被 `j` 移出一次，因此内层 `while` 的总执行次数也是 $O(n)$。

## 复杂度
- 时间复杂度：$O(n)$，两个指针各移动 $n$ 次
- 空间复杂度：$O(1)$，哈希表仅存储三种字符的计数

# 解题思路（记录最近位置）
滑动窗口的核心思路是"对每个 `j` 找最小 `i`"。换个角度，我们可以**对每个 `i` 找最大的 `j`**。

维护三个变量 `last_a`、`last_b`、`last_c`，分别记录字符 `'a'`、`'b'`、`'c'` 最近一次出现的下标（初始为 `-1`）。

当遍历到位置 `i` 时，以 `i` 为**右端点**的合法子串，其左端点 $j$ 必须满足：$j \le \min(last_a, last_b, last_c)$——因为 $(j, i]$ 需要包含三种字符，左端点必须**不晚于**三种字符中最早出现者的位置。

由于 $j$ 可以从 $0$ 取到 $\min(last_a, last_b, last_c)$，共有 $\min(last_a, last_b, last_c) + 1$ 个（`+1` 是因为 $j=0$ 也算一个）。

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        ans = 0
        last = {'a': -1, 'b': -1, 'c': -1}
        for i, c in enumerate(s):
            last[c] = i
            ans += min(last['a'], last['b'], last['c']) + 1
        return ans
```

以 `s = "abcabc"` 跟踪：

| `i` | `c` | `last` | `min(last)+1` | `ans` |
|:--:|:--:|:--|:--:|:--:|
| 0 | `a` | {a:0, b:-1, c:-1} | 0 | 0 |
| 1 | `b` | {a:0, b:1, c:-1} | 0 | 0 |
| 2 | `c` | {a:0, b:1, c:2} | 1 | 1 |
| 3 | `a` | {a:3, b:1, c:2} | 2 | 3 |
| 4 | `b` | {a:3, b:4, c:2} | 3 | 6 |
| 5 | `c` | {a:3, b:4, c:5} | 4 | 10 |

同样得到 `10`。

这个写法的代码极简，但思维跳跃较大——需要理解"以 `i` 结尾的子串数量 = 最早出现位置 + 1"这个关系。面试中建议两种都掌握，滑动窗口更直观，最近位置更优雅。

| 对比项 | 滑动窗口 (`ans += n - i`) | 最近位置 (`ans += min(last) + 1`) |
| --- | --- | --- |
| 时间 | $O(n)$ | $O(n)$ |
| 空间 | $O(1)$ | $O(1)$ |
| 代码行数 | 13 行 | 7 行 |
| 直观程度 | 较直观 | 需要一点推导 |
| 适合场景 | 需要明确维护窗口时 | 追求极简代码时 |

# 代码优化
两种 $O(n)$ 算法都已经是最优——至少需要遍历一次字符串。以下优化聚焦于代码细节。

## 滑动窗口：用普通字典替代 `defaultdict`
本题只需三个键 `'a'`、`'b'`、`'c'`，初始化时直接给出初始值可以省去 `collections` 导入：

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ans = j = 0
        d = {'a': 0, 'b': 0, 'c': 0}

        for i, c in enumerate(s):
            d[c] += 1
            while d['a'] and d['b'] and d['c']:
                ans += n - i
                d[s[j]] -= 1
                j += 1

        return ans
```

写法和 `defaultdict` 完全等价，但减少了一次 `import`。

## 最近位置：用三个独立变量
`min(last['a'], last['b'], last['c'])` 每次都要做三次字典查找和两次 `min` 比较。用三个独立整型变量可以减少字典开销（虽然在这个数据规模下差异微乎其微）：

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        ans = 0
        last_a = last_b = last_c = -1
        for i, c in enumerate(s):
            if c == 'a':
                last_a = i
            elif c == 'b':
                last_b = i
            else:
                last_c = i
            ans += min(last_a, last_b, last_c) + 1
        return ans
```

用 `if/elif/else` 取代字典赋值，常数更小；`min` 三个独立变量也比字典取值略快。在算法竞赛或极致性能场景下可选用此写法，但日常使用中字典版本的可读性更好。
