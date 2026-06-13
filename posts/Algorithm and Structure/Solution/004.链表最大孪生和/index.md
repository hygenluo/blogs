---
title: 链表最大孪生和
author: Hygen
description: LeetCode-2130
category: Algorithm and Structure
tags:[Solution, 双指针, 链表, DFS]
published: 2026-06-14
---
# 题目描述
[题目链接](https://leetcode.cn/problems/maximum-twin-sum-of-a-linked-list/description/?envType=daily-question&envId=2026-06-14)

在一个大小为`n`且`n`为偶数的链表中，第1个节点和第n个节点为一对孪生节点，第2对与第n-1对为一对孪生节点，依此类推...
**孪生和**为一个节点和它的孪生节点两者之和

给一个长度为`n`的链表`head`，要求返回链表的最大孪生和

## 示例1
> input: head=[5， 4， 2， 1]
> output: 6
