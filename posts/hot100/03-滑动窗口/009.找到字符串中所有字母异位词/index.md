---
title: 找到字符串中所有字母异位词
author: Hygen
description: Leetcode-438
category: hot100
tags: [Solution, 滑动窗口]
published: 2026-06-28
---
# 题目描述
[题目链接](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked)

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的**字母异位词**的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**字母异位词**指字母相同，但排列不同的字符串。

## 输入
字符串 `s` 和 `p`
## 输出
所有异位词子串在 `s` 中的起始下标列表
## 约束条件
1. `1 <= s.length, p.length <= 3 * 10^4`
2. `s` 和 `p` 仅包含小写英文字母
# 进阶约束
1. 能否将时间复杂度优化到 $O(n)$？
## 示例1
```
input: s = "cbaebabacd", p = "abc"
output: [0, 6]
```
解释：起始下标 `0` 的子串是 `"cba"`，下标 `6` 的子串是 `"bac"`，二者都是 `"abc"` 的字母异位词。
## 示例2
```
input: s = "abab", p = "ab"
output: [0, 1, 2]
```
解释：起始下标 `0` 的子串是 `"ab"`，下标 `1` 的子串是 `"ba"`，下标 `2` 的子串是 `"ab"`。
# 解题思路
题目要求找 `s` 中所有**连续**子串，使其与 `p` 互为字母异位词——即长度相同、各字符出现次数完全一致。暴力枚举每个起点再排序或逐字符比对，在 $n$ 可达 $3 \times 10^4$ 时会超时。能否在向右扩展子串的同时，**动态维护**「当前窗口的字符频次是否合法」？

## 从暴力枚举出发
最直观的做法：枚举每个起点 `i`，取长度为 `len(p)` 的子串，判断它是否与 `p` 互为异位词。判断方式可以是排序后比较，或用 `Counter` 比较频次：

```python
from collections import Counter

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        m = len(p)
        cp = Counter(p)
        for i in range(len(s) - m + 1):
            if Counter(s[i:i + m]) == cp:
                res.append(i)
        return res
```

每个起点都要构造一次子串频次，时间复杂度 $O(n \cdot m)$，数据规模下会超时。瓶颈在于**相邻起点的大量重复计算**——窗口右移一位时，只是去掉最左字符、加入最右字符，频次可以增量更新。能否让左右边界只各移动一次？

## 滑动窗口：固定长度 + 频次比较
用 `[j, i]` 维护一个窗口。右指针 `i` 每步将 `s[i]` 的计数 `+1`；当窗口长度超过 `len(p)` 时，将 `s[j]` 的计数 `-1` 并 `j += 1`，保持窗口长度恰好为 `len(p)`。每次窗口就位后，比较窗口频次与 `p` 的频次是否相等：

```python
from collections import Counter, defaultdict

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m = len(p)
        cp = Counter(p)
        cs = defaultdict(int)
        res = []
        j = 0
        for i, c in enumerate(s):
            cs[c] += 1
            if i - j + 1 > m:
                cs[s[j]] -= 1
                j += 1
            if i - j + 1 == m and cs == cp:
                res.append(j)
        return res
```

每次比较两个字典是 $O(|\Sigma|)$，$|\Sigma| = 26$ 视为常数，总时间 $O(n)$。但每次都要做完整字典相等判断，常数偏大——能否**只在不合法时收缩**，而不是每步都全量比较？

## 滑动窗口 + 计数：超频则左缩
思路与「无重复字符的最长子串」的计数写法一脉相承：右指针 `i` 每步将 `s[i]` 计数 `+1`；若某字符在窗口内的出现次数**超过了** `p` 中该字符的次数（`cs[c] > cp[c]`），说明当前窗口不可能成为异位词，用 `while` 从左侧逐个减计数并右移 `j`，直到该字符不再超频。收缩完毕后，若窗口长度恰好等于 `len(p)`，则各字符频次必然与 `p` 完全一致（总数相同且无人超频），记录起点 `j`：

```python
from collections import Counter, defaultdict

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        j = 0
        cs = defaultdict(int)
        cp = Counter(p)
        res = []

        for i, c in enumerate(s):
            cs[c] += 1
            while cs[c] > cp[c]:
                cs[s[j]] -= 1
                j += 1
            if i - j + 1 == len(p):
                res.append(j)
        
        return res
```

以 `s = "cbaebabacd"`、`p = "abc"` 为例：

| `i` | `c` | 操作 | 窗口 `[j, i]` | 是否记录 |
| --- | --- | --- | --- | --- |
| 0 | `c` | 加入 `c` | `[0,0]` 长度 1 | — |
| 1 | `b` | 加入 `b` | `[0,1]` 长度 2 | — |
| 2 | `a` | 加入 `a` | `[0,2]` = `"cba"` 长度 3 | 记录 `0` |
| 3 | `e` | `e` 不在 `p` 中，`cs['e']=1 > cp['e']=0`，左缩至 `j=4` | 长度 0 | — |
| 4 | `b` | 加入 `b` | `[4,4]` 长度 1 | — |
| 5 | `a` | 加入 `a` | `[4,5]` = `"ba"` 长度 2 | — |
| 6 | `b` | `cs['b']=2 > cp['b']=1`，左缩至 `j=5` | `[5,6]` = `"ab"` 长度 2 | — |
| 7 | `a` | `cs['a']=2 > cp['a']=1`，左缩至 `j=6` | `[6,7]` = `"ba"` 长度 2 | — |
| 8 | `c` | 加入 `c` | `[6,8]` = `"bac"` 长度 3 | 记录 `6` |

最终答案为 `[0, 6]`。

`j` 只增不减，每个字符最多入窗、出窗各一次，时间复杂度 $O(n)$。

## 复杂度
- 时间复杂度：$O(n)$，`i` 与 `j` 各最多移动 $n$ 次
- 空间复杂度：$O(|\Sigma|)$，哈希表最多存字符集大小个键；本题仅小写字母，视为 $O(1)$

# 解题思路（固定窗口 + 匹配计数）
上一节的「超频则左缩」是**变长窗口**：只有出现非法字符时才收缩，逻辑简洁，且与「至多出现 $k$ 次」类题目共用同一模板。本题还有一个常见写法：**固定窗口长度**始终为 `len(p)`，用「匹配度」代替字典全量比较。

维护 `need = Counter(p)` 记录 `p` 中各字符的目标频次，`window` 记录当前窗口频次，再用 `valid` 统计「有多少种字符的窗口计数**恰好等于**目标计数」。右扩时若某字符计数从小于变为等于目标，`valid += 1`；若从等于变为大于目标，`valid -= 1`。窗口超长时左缩一位，对离开窗口的字符做对称更新。当 `valid == len(need)`（所有出现过的字符类型都匹配），当前起点即为异位词：

```python
from collections import Counter, defaultdict

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s) < len(p):
            return []
        need = Counter(p)
        window = defaultdict(int)
        valid = 0
        res = []
        j = 0
        m = len(p)

        for i, c in enumerate(s):
            window[c] += 1
            if c in need:
                if window[c] == need[c]:
                    valid += 1
                elif window[c] == need[c] + 1:
                    valid -= 1

            if i - j + 1 > m:
                left = s[j]
                if left in need:
                    if window[left] == need[left]:
                        valid -= 1
                    elif window[left] == need[left] + 1:
                        valid += 1
                window[left] -= 1
                j += 1

            if valid == len(need):
                res.append(j)

        return res
```

两种写法时间均为 $O(n)$，但侧重点不同：

| 对比项 | 超频则左缩（变长窗口） | 匹配计数（固定窗口） |
| --- | --- | --- |
| 时间 | $O(n)$ | $O(n)$ |
| 内层循环 | 有 `while` | 无 |
| 合法性判断 | 任一字符超频即收缩 | `valid` 达标即合法 |
| 扩展性 | 易改为「至多 / 至少 $k$ 次」 | 适合「频次恰好相等」类题 |

面试中若强调**通用模板**，推荐匹配计数写法；若刚学完「无重复字符的最长子串」，超频左缩写法更连贯。

# 代码优化
算法层面两种 $O(n)$ 思路都已最优（至少需读一遍 `s`）。以下优化侧重**常数因子**与**代码简洁度**。

## 用长度 26 的数组替代 `Counter`
题目保证只有小写字母，频次用 `ord(c) - ord('a')` 映射到下标 `0~25`，比 `Counter` / `defaultdict` 更省开销：

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m = len(p)
        if len(s) < m:
            return []
        cp = [0] * 26
        cs = [0] * 26
        for c in p:
            cp[ord(c) - 97] += 1

        res = []
        j = 0
        for i, c in enumerate(s):
            idx = ord(c) - 97
            cs[idx] += 1
            while cs[idx] > cp[idx]:
                cs[ord(s[j]) - 97] -= 1
                j += 1
            if i - j + 1 == m:
                res.append(j)
        return res
```

语义与原写法完全一致，访问数组比哈希表更快，空间也固定为 $O(26)$。

## 匹配计数写法：用 `diff` 数组判断全匹配
固定窗口写法中，不必维护 `valid` 变量——用一个 `diff[26]` 记录「窗口计数与目标计数的差」，窗口就位后检查 `diff` 是否全零即可：

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m = len(p)
        n = len(s)
        if n < m:
            return []
        diff = [0] * 26
        for c in p:
            diff[ord(c) - 97] += 1

        res = []
        for i, c in enumerate(s):
            diff[ord(c) - 97] -= 1
            if i >= m:
                diff[ord(s[i - m]) - 97] += 1
            if i >= m - 1 and not any(diff):
                res.append(i - m + 1)
        return res
```

`any(diff)` 最坏 $O(26)$，仍是 $O(n)$。若追求极致常数，可像 `valid` 写法一样增量维护「差值为零的字符种类数」，避免每步扫描 26 个位置。
