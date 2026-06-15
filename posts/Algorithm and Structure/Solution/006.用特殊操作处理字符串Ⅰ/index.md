---
title: 用特殊操作处理字符串Ⅰ
author: Hygen
description: LeetCode-3612
category: Algorithm and Structure
tags: [Solution, 字符串, 模拟]
published: 2026-06-16
---
# 题目描述
[题目链接](https://leetcode.cn/problems/process-string-with-special-operations-i/description/?envType=daily-question&envId=2026-06-16)

给你一个字符串 `s`，它由小写英文字母和特殊字符 `*`、`#` 和 `%` 组成。

请根据以下规则从左到右处理 `s` 中的字符，构造一个新的字符串 `result`：

- 如果字符是**小写**英文字母，则将其添加到 `result` 中。
- 字符 `'*'` 会**删除** `result` 中的最后一个字符（如果存在）。
- 字符 `'#'` 会**复制**当前的 `result` 并**追加**到其自身后面。
- 字符 `'%'` 会**反转**当前的 `result`。

在处理完 `s` 中的所有字符后，返回最终的字符串 `result`。

## 输入
字符串 `s`
## 输出
按规则处理后的字符串 `result`
## 约束条件
1. `1 <= s.length <= 20`
2. `s` 只包含小写英文字母和特殊字符 `*`、`#` 和 `%`
## 示例1
```
input: s = "a#b%*"
output: "ba"
```
解释：依次得到 `"a"` → `"aa"` → `"aab"` → `"baa"` → `"ba"`
## 示例2
```
input: s = "z*#"
output: ""
```
解释：依次得到 `"z"` → `""` → `""`
# 解题思路
题目要求从左到右依次处理每个字符，并根据当前字符对 `result` 做追加、删除、复制或反转。这类题没有更巧妙的数学转化，核心就是**按题意模拟**。

模拟时需要维护一个可变的中间结果。若直接用字符串拼接，遇到 `'*'` 删除末尾、`'%'` 反转时都要频繁切片或重建，写法繁琐。更自然的做法是使用列表 `res` 作为当前 `result` 的字符容器：

- 普通字母：直接 `append`，等价于追加到末尾。
- `'*'`：若 `res` 非空则 `pop()`，删除最后一个字符。
- `'#'`：`res += res`，将当前内容整体复制一份拼到后面。
- `'%'`：`res.reverse()`，原地反转。

遍历结束后 `''.join(res)` 即为答案。列表尾部增删是 $O(1)$，复制和反转与当前结果长度有关；在题目给定的数据规模下完全足够。

## 代码
```python
class Solution:
    def processStr(self, s: str) -> str:
        res = []

        for c in s:
            if c == '*': 
                if res: res.pop()
            elif c == '#': res += res
            elif c == '%': res.reverse()
            else: res.append(c)

        return ''.join(res)
```
