# 博客标签与分类目录

## 现有标签 (Tags)

### LeetCode / 算法标签
`Solution` — 所有算法题解统一使用的标签（必加）

**数据结构类：**
`哈希表` `链表` `栈` `单调栈` `单调队列` `字符串` `数组` `矩阵` `二叉树` `堆` `前缀树` `并查集`

**算法技巧类：**
`双指针` `滑动窗口` `前缀和` `DFS` `BFS` `动态规划` `贪心` `回溯` `二分查找` `排序` `数学` `模拟` `位运算` `分治` `递归` `记忆化搜索`

**标签复用原则：**
- 每题必须包含 `Solution` 标签
- 根据解题实际使用的核心技术选 1-3 个技术标签
- 优先使用已存在的标签名（如用 `双指针` 而不是 `Two Pointers`）
- 若确实没有合适的现有标签，可以新建，但**必须经用户同意**后方可使用

### AI / Agent 标签
`Agent` `LangGraph` `LLM` `RAG` `NLP` `Transformer` `ML` `Deep Learning`

### 通用标签
`guide` — 教程/指南类
`FastAPI` — FastAPI 相关

## 现有分类 (Categories)

| Category | 用途 | 对应文件夹 |
|----------|------|-----------|
| `hot100` | LeetCode Hot100 题目 | `posts/hot100/` |
| `Algorithm and Structure` | 算法与数据结构题解（非 hot100） | `posts/Algorithm and Structure/Solution/` |
| `Agent` | LangGraph/Agent 相关 | `posts/Agent/` |
| `Guides` | 通用教程/指南 | `posts/guides/` |

**注意：** PyLearning、NLP、Learning 等文件夹中的文章当前**未使用** `category` 字段。若在这些文件夹创建新文章，可酌情添加 category 或保持不填。

## Frontmatter 参考

### 必填字段
- `title` — 文章标题
- `published` — 发布日期 (YYYY-MM-DD)
- `description` — 简短描述（用于 SEO 和预览卡片）

### 常用可选字段
- `author: Hygen` — 固定作者名
- `tags: [tag1, tag2]` — 标签数组
- `category: xxx` — 分类
- `draft: false` — 是否为草稿
- `image: ./cover.png` — 封面图（相对路径）

### 字段注意事项
- `author` 始终为 `Hygen`
- `published` 使用当天日期
- `draft` 默认为 `false`
- 不要使用 `pubDate`（旧字段，已被 `published` 取代）
- 不要使用 `licenseName`（除非有特殊需求）
