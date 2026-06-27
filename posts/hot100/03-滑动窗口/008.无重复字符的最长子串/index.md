---
title: 无重复字符的最长子串
author: Hygen
description: Leetcode-3
category: hot100
tags: [Solution, 滑动窗口]
published: 2026-06-28
---
# 题目描述
[题目链接](https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked)

给定一个字符串 `s`，请你找出其中不含有重复字符的**最长子串**的长度。

## 输入
字符串 `s`
## 输出
最长无重复子串的长度（整数）
## 约束条件
1. `0 <= s.length <= 5 * 10^4`
2. `s` 由英文字母、数字、符号和空格组成
# 进阶约束
1. 能否将时间复杂度优化到 $O(n)$？
## 示例1
```
input: s = "abcabcbb"
output: 3
```
解释：最长无重复子串为 `"abc"`，长度为 `3`。
## 示例2
```
input: s = "bbbbb"
output: 1
```
解释：最长无重复子串为 `"b"`，长度为 `1`。
## 示例3
```
input: s = "pwwkew"
output: 3
```
解释：最长无重复子串为 `"wke"`，长度为 `3`。注意 `"pwke"` 是子序列而非子串。
# 解题思路
题目要求找**连续**子串，且子串内每个字符最多出现一次。暴力枚举所有子串再逐个判重，在 $n$ 可达 $5 \times 10^4$ 时会超时——能否在扩展子串的同时，**动态维护**「当前窗口是否合法」？

## 从暴力枚举出发
最直观的做法：枚举所有起点 `i` 和终点 `j`，检查 `s[i:j+1]` 是否有重复字符：

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        n = len(s)
        for i in range(n):
            seen = set()
            for j in range(i, n):
                if s[j] in seen:
                    break
                seen.add(s[j])
                ans = max(ans, j - i + 1)
        return ans
```

内层遇到重复就 `break`，每个起点最多扫到第一次冲突为止。最坏情况（如全不重复）时间复杂度 $O(n^2)$，仍可能超时。瓶颈在于：**不同起点之间大量重复扫描**——能否让左右边界只各移动一次？

## 滑动窗口：右扩左缩
用两个指针维护一个**窗口** `[j, i]`（闭区间）：

- `i`：右边界，逐字符向右扩展
- `j`：左边界，当窗口内出现重复时向右收缩

再用一个集合 `seen` 记录窗口内有哪些字符。`i` 每右移一步，若 `s[i]` 不在集合中则加入并更新答案；若已存在，则不断将 `s[j]` 从集合移除并 `j += 1`，直到可以安全放入 `s[i]`：

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = j = 0
        seen = set()
        for i, c in enumerate(s):
            while c in seen:
                seen.remove(s[j])
                j += 1
            seen.add(c)
            ans = max(ans, i - j + 1)
        return ans
```

以 `s = "pwwkew"` 为例：

| `i` | `c` | 窗口 `[j, i]` | 操作 | `ans` |
| --- | --- | --- | --- | --- |
| 0 | `p` | `[0,0]` | 加入 `p` | 1 |
| 1 | `w` | `[0,1]` | 加入 `w` | 2 |
| 2 | `w` | — | `w` 重复，移除 `p`、`w`（`j` 到 2） | 2 |
| 2 | `w` | `[2,2]` | 加入 `w` | 2 |
| 3 | `k` | `[2,3]` | 加入 `k` | 2 |
| 4 | `e` | `[2,4]` | 加入 `e` | 3 |
| 5 | `w` | — | `w` 重复，移除 `w`（`j` 到 3） | 3 |
| 5 | `w` | `[3,5]` | 加入 `w` | 3 |

最终答案为 `3`（`"wke"`）。

`j` 只增不减，每个字符最多入窗、出窗各一次，时间复杂度 $O(n)$。

## 滑动窗口 + 计数：用出现次数判重
集合的 `c in seen` 等价于「该字符出现次数 $\ge 1$」。改用哈希表 `d` 统计窗口内各字符出现次数，逻辑可以写得更统一：

1. 右指针 `i` 每步将 `s[i]` 的计数 `+1`
2. 若某字符计数 $> 1$，说明窗口非法，用 `while` 从左侧逐个减计数并右移 `j`，直到该字符计数回到 `1`
3. 窗口合法后，用 `i - j + 1` 更新答案

```python
from collections import defaultdict

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = j = 0
        d = defaultdict(int)
        for i, c in enumerate(s):
            d[c] += 1
            while d[c] > 1:
                d[s[j]] -= 1
                j += 1
            ans = max(ans, i - j + 1)
        
        return ans
```

以 `s = "abcabcbb"` 为例，当 `i = 3`、`c = 'a'` 时：

| 步骤 | `d['a']` | `j` | 窗口 | `ans` |
| --- | --- | --- | --- | --- |
| 加入 `a`（`i=3`） | 2 | 0 | 非法 | 3 |
| `d[s[0]]-=1`，`j=1` | 1 | 1 | `[1,3]` = `"bca"` | 3 |

此时 `'a'` 计数恢复为 `1`，窗口合法，长度为 `3`。

与集合写法相比，计数写法把「是否存在」推广为「出现几次」，后续遇到**允许字符出现 $k$ 次**的滑动窗口变形题时可以直接复用同一套框架。

## 复杂度
- 时间复杂度：$O(n)$，`i` 与 `j` 各最多移动 $n$ 次
- 空间复杂度：$O(|\Sigma|)$，哈希表最多存字符集大小个键；字符集有限时视为 $O(1)$

# 解题思路（直接跳转左边界）
计数写法中，`while d[c] > 1` 会**逐字符**收缩左边界。其实重复发生时，左边界不必慢慢挪——只需跳到**上一次该字符出现位置的下一格**。

维护 `last[c]` 为字符 `c` 最近一次出现的下标。当在位置 `i` 再次遇到 `c` 时，若 `last[c] >= j`（说明 `c` 仍在当前窗口内），则直接令 `j = last[c] + 1`；否则 `j` 不动（旧位置已被之前的收缩甩在窗口外）。然后更新 `last[c] = i` 并刷新答案：

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}
        ans = j = 0
        for i, c in enumerate(s):
            if c in last and last[c] >= j:
                j = last[c] + 1
            last[c] = i
            ans = max(ans, i - j + 1)
        return ans
```

以 `s = "abba"` 为例：当 `i = 3`、`c = 'a'` 时，`last['a'] = 0` 且 `0 >= j`，直接 `j = 1`，窗口变为 `"ba"`，无需经过 `"bba"` 的中间状态。

两种写法时间均为 $O(n)$，但跳转写法每次循环只做常数操作，没有内层 `while`，常数更小，代码也更短。

| 对比项 | 计数 + 逐位收缩 | 记录下标 + 直接跳转 |
| --- | --- | --- |
| 时间 | $O(n)$ | $O(n)$ |
| 内层循环 | 有 `while` | 无 |
| 扩展性 | 易改为「最多出现 $k$ 次」 | 需额外维护次数 |
| 适用 | 通用滑动窗口模板 | 本题「至多 1 次」特化 |

面试中若强调**通用模板**，推荐计数写法；若只求本题最简实现，记录下标跳转更利落。

# 代码优化
算法层面两种 $O(n)$ 思路都已最优（至少需读一遍字符串）。以下优化侧重**代码简洁度**与**常数细节**。

## 用普通字典替代 `defaultdict`
本题只需「先加再减」，完全可以用 `dict.get(key, 0)` 或 `d[c] = d.get(c, 0) + 1`，少一个 `collections` 依赖：

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = j = 0
        d = {}
        for i, c in enumerate(s):
            d[c] = d.get(c, 0) + 1
            while d[c] > 1:
                d[s[j]] -= 1
                j += 1
            ans = max(ans, i - j + 1)
        return ans
```

语义与 `defaultdict(int)` 完全一致。

## 跳转写法：合并更新与判断
记录下标版本中，`last[c] = i` 可以放在判断之前，用 `max` 一行完成边界跳转，减少分支嵌套：

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}
        ans = j = 0
        for i, c in enumerate(s):
            j = max(j, last.get(c, -1) + 1)
            last[c] = i
            ans = max(ans, i - j + 1)
        return ans
```

`last.get(c, -1) + 1` 在 `c` 首次出现时等于 `0`，与 `j` 取 `max` 后仍保持 `j` 不变，等价于原来的 `if c in last and last[c] >= j` 判断。
