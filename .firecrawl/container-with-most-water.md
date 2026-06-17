![leetcode](https://static.leetcode.cn/cn-frontendx-assets/production/_next/static/images/logo-711e116152be014f445f50aa6a369231.png)在力扣 App 中打开

题目描述

题目描述

题解

题解

提交记录

提交记录

3

代码

代码

1

测试用例

测试用例

测试结果

测试结果

2

[11\. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

中等

相关标签

![premium lock icon](https://static.leetcode.cn/cn-frontendx-assets/production/_next/static/images/lock-a6627e2c7fa0ce8bc117c109fb4e567d.svg)相关企业

提示

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：** 你不能倾斜容器。

**示例 1：**

![](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

**示例 2：**

```
输入：height = [1,1]
输出：1
```

**提示：**

- `n == height.length`
- `2 <= n <= 105`
- `0 <= height[i] <= 104`

通过次数

2,219,368/3.6M

通过率

61.8%

* * *

相关标签

[贪心](https://leetcode.cn/tag/greedy/) [数组](https://leetcode.cn/tag/array/) [双指针](https://leetcode.cn/tag/two-pointers/)

* * *

![icon](https://static.leetcode.cn/cn-frontendx-assets/production/_next/static/images/lock-a6627e2c7fa0ce8bc117c109fb4e567d.svg)

相关企业

* * *

提示 1

If you simulate the problem, it will be O(n^2) which is not efficient.

* * *

提示 2

Try to use two-pointers. Set one pointer to the left and one to the right of the array. Always move the pointer that points to the lower line.

* * *

提示 3

How can you calculate the amount of water at each step?

* * *

相似题目

[接雨水](https://leetcode.cn/problems/trapping-rain-water/)

困难

[礼盒的最大甜蜜度](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/)

中等

[打家劫舍 IV](https://leetcode.cn/problems/house-robber-iv/)

中等

* * *

评论 (0)

评论

💡 讨论区规则

1\. 请不要在评论区发表题解！

2\. 评论区可以发表关于对翻译的建议、对题目的疑问及其延伸讨论。

3\. 如果你需要整理题解思路，获得反馈从而进阶提升，可以去题解区进行。

排序:最热

* * *

贡献者![](https://static.leetcode-cn.com/cn-legacy-assets/images/LeetCode_avatar.png)

© 2026 领扣网络（上海）有限公司

6K

0

行 1，列 1

运行和提交代码需要 [登录](https://leetcode.cn/accounts/login/?next=%2Fproblems%2Fcontainer-with-most-water%2Fdescription%2F%3FenvType%3Dstudy-plan-v2%26envId%3Dtop-100-liked)

Case 1Case 2

height =

\[1,8,6,2,5,4,8,3,7\]

9

1

2

›

\[1,8,6,2,5,4,8,3,7\]

\[1,1\]

Source

FindHeaderBarSize

FindTabBarSize

FindBorderBarSize