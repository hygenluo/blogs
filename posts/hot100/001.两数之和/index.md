---
title: 两数之和
author: Hygen
description: Leetcode-1
category: hot100
tags: [Solution, 哈希表]
published: 2026-06-14
---
# 题目描述
[题目链接](https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked)
## 输入
一个整数数组`nums`

一个整数目标值`target`
## 输出
两个数组下标，数组中这两个下标的值的和为`target`
## 约束条件
1. 不能使用两次相同的下标
2. 可以按任意顺序返回答案
3. 只存在一个有效答案
# 进阶约束
1. 时间复杂度需要小于$O(n^2)$
## 示例1
```
input: nums = [2,7,11,15], target = 9
output: [0,1]
```
解释： nums[0] = 2, nums[1] = 7, 2 + 7 = 9
## 示例2
```
input: nums = [3,2,4], target = 6
output: [1,2]
```
## 示例3
```
input: nums = [3,3], target = 6
output: [0,1]
```
# 解题思路
题目很好理解，从数组中找出两个值，和为`target`即可，然后返回这两个值的下标就行。

很容易想到使用双循环来完成这题，没什么思考难度，不多赘述。但是进阶约束中需要我们的时间复杂度小于$O(n^2)$，这显然就是为了卡双循环，因此此路不通。不过我们依旧可以先将这个暴力解法的代码写出来，看看能不能优化：
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
```

从暴力的第二层循环来看，它并不在乎以前的值，因为以前的值在第一层循环中已经经历过了，比如当`i = 2, j = 3`时，`i = 1`的情况在第一层循环中已经结束了，因此返过来，对于第二层循环来说，都是每个`nums[j]`值在前面已经遍历过的数中找和为`target`的配对，从这个角度上看，第一层循环就成为了非必要的循环，只要我们能找到一种方法，能够直接将以前遍历过的数存起来就行了。

什么方法能直接存以前遍历过的数值呢？显然`哈希表`是可以的，且它的存取都是$O(1)$的，这不需要自己动手造轮子，直接使用python中的`defaultdict`类即可。

## 代码
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hh = defaultdict(int)
        for i, x in enumerate(nums):
            if target - x in hh: return [hh[target - x], i]
            hh[x] = i
```