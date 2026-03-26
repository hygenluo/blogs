---
title: FastAPI入门实战
description: 简单入门及实战
author: Hygen
tags: [FastAPI, guide]
published: 2025-12-17
---
# 1. FastAPI入门
## 1.1. 什么是FastAPI
`FastAPI` 是高性能的 `Python Web` 框架，专门用于构建 `API（应用程序接口）`。它基于标准的 Python `类型提示（type hints）`，是 Python 领域最受欢迎的 Web 框架之一。
## 1.2. FastAPI有什么优点
1. `高性能`：可大量处理并发请求
2. `高效率`：类型检查，减少人为错误，开发速度可提升200%~300%
3. `交互式`：自动生成Swagger UI文档(/docs)和ReDoc文档
# 2. 入门实战
## 2.1. 准备工作
**准备虚拟环境**

此处使用`uv`来进行包管理。

1. 使用`uv init fastapi-demo`来初始化一个fastapi项目，进入项目文件夹
2. 添加`fastapi`库：`uv add fastapi uvicorn[standard]`

## 2.2. 第一个FastAPI项目
在`main.py`中输入如下代码：
```python
from fastapi import FastAPI

# 注册FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = "query"):
    return {"item_id": item_id, "q": q}
```
在命令行中输入`uv run uvicorn main:app --reload`来运行第一个项目，在浏览器中输入`http://127.0.0.1:8000/`来访问这个项目，出现如下内容说明已经成功运行：

![first running](./ac7514e3-222d-4362-87a9-cd0077cfb261.png)

要访问api文档只需要在链接后面添加`/docs`就好了，完整链接：`http://127.0.0.1:8000/docs`，内容如下：

![docs](./ada70338-ec49-4d1e-92ee-2f453f1b202d.png)

## 2.3. 路由
路由简单来说就是**URL地址**和**处理函数**之间的映射关系，它决定了用户访问某个网址时，服务器应该执行哪段代码返回结果。
### 路由实战
> 需求：当访问路径`/user/hello`时，服务器的响应结果是`{"msg":"Hello, Zako"}`

#### 代码
main.py
```python
@app.get("/user/hello")
def hello():
    return {"message": "Hello, Zako!"}
```
运行结果和上面类似，不再贴出
## 2.4. 参数
参数就是客户端发送请求时**额外附带的信息和指令**

参数的作用是让同一个接口能根据不同的输入，返回不同的输出，实现**动态交互**

参数分为三类，`路径参数`, `查询参数`, `请求体`
### 2.4.1. 路径参数
![路径参数](./e2dd32ab-bc36-4821-9f43-45f19de8a447.png)

#### 路径参数实战01
> 需求：以用户id为路径参数设计URL，要求响应结果包含用户id和名称（yusu+id）

样例：
```
URL： /user/114514
响应结果：{id: 114514, name: yusu114514}
```
##### 代码
main.py
```python
@app.get("/user/{id}")
def get_user(id: int):
    return {"id": id, "name": f"yusu{id}"}
```
#### 路径参数实战02
> 需求： 定义两个接口，携带路径参数，并使用 Path 来实现类型注解
>
> 具体如下：
> 
> 接口1： 以新闻分类id为参数设计URL，id范围为1~100
>
> 接口2： 以新闻分类名称为参数设计URL，分类名称长度为2~10
##### 代码
main.py
```python
@app.get("/news/{id}")
def get_news(id: int = Path(..., ge=1, le=100)):
    return {"id": id, "title": f"news{id}"}

@app.get("/news/{name}")
def get_name(name: str = Path(..., min_length=2, max_length=10)):
    return {"name": name}
```
### 2.4.2. 查询参数
![查询参数](./02f293fd-b097-4596-b6fb-14eee1048746.png)
#### 查询参数实战
> 需求：设计接口查询图书，要求携带两个查询参数：图书分类和价格
>
> 参数具体要求：
>
> 图书分类：默认值为 Python开发，长度限制5 ~ 255
>
> 价格：限制大小范围 50 ~ 100
##### 代码
main.py
```python
@app.get("/books")
def get_books(category: str = Query(default="Python开发", min_length=5, max_length=255), price: float = Query(ge=50, le=100)):
    return {"category": category, "price": price}
```
### 2.4.3. 请求体
![请求体](./5c636822-470f-4b83-9267-64e73478cdd5.png)
#### 请求体实战01
> 需求：设计接口新增图书，图书信息包含：书名、作者、出版社、售价
##### 代码
main.py
```python
class Book(BaseModel):
    title: str
    author: str
    publisher: str
    price: float

@app.post("/books")
def create_book(book: Book):
    return {"book": book}
```
#### 请求体实战02
> 需求：设计接口新增图书，图书信息包含：书名、作者、出版社、售价
>
> 具体要求如下：
>
> 书名：不能为空；长度 2 ~ 20
>
> 作者：长度 2 ~ 10
>
> 出版社：默认值“黑马出版社”
>
> 售价：不能为空；价格大于0元
##### 代码
main.py
```python
class Books(BaseModel):
    title: str = Field(..., min_length=2, max_length=20)
    author: str = Field(..., min_length=2, max_length=10)
    publisher: str = Field(default="黑马出版社")
    price: float = Field(..., gt=0)
@app.post("/book")
def create_book(book: Books):
    return {"book": book}
```
## 2.4. 响应
既然用户对服务器有请求，那么服务器自然需要有响应。
### 2.4.1. 响应类型
FastAPI 默认会将你路径操作函数返回的任何数据自动转换为 JSON，并使用`JSONResponse`作为默认响应。同时FastAPI也提供了丰富的响应类型和自定义的返回类型来返回不同的数据。
| 响应类型          | 用途                 | 示例                          |
| ----------------- | -------------------- | ----------------------------- |
| JSONResponse      | 默认响应，返回JSON数据 | return {"key": "value"}       |
| HTMLResponse      | 返回HTML内容          | return HTMLResponse(html_content) |
| PlainTextResponse | 返回纯文本            | return PlainTextResponse("text") |
| FileResponse      | 返回文件下载          | return FileResponse(path)     |
| StreamingResponse | 流式响应             | 生成器函数返回数据            |
| RedirectResponse  | 重定向               | return RedirectResponse(url)  |
其中最常用的是`JSON`、`HTML`、`File`格式。可以通过声明`response_class`参数或直接返回一个`Response`对象来覆盖默认行为，定制自己想要的响应类型。
### 2.4.2. 响应类型设置方式
![响应类型设置方式](./982384ac-64e6-4137-ba8f-c1b9cb51aef2.png)
### 2.4.3. 自定义响应类型
`response_model`是路径操作装饰器（如 @app.get或 @app.post）的关键参数，它通过一个`Pydantic`模型来严格定义和约束 API 端
点的输出格式。这一机制在提供自动数据验证和序列化的同时，更是保障数据安全性的第一道防线。 
```python
from pydantic import BaseModel
class News(BaseModel):
    id: int
    title: str
    content: str

@app.get("/news/{id}", response_model=News)
async def get_news(id: int):
    return {
        "id": id,
        "title": f"这是第{id}本书",
        "content": "这是一本好书"
}
```
### 2.4.4. 异常处理
对于客户端引发的错误（4xx，如资源未找到、认证失败），应使用`fastapi.HTTPException`来中断正常处理流程，
并返回标准错误响应 。
```python
from fastapi import FastAPI, HTTPException
@app.get('/news/{id}')
async def get_news(id: int):
    id_list = [1, 2, 3, 4, 5, 6]
    if id not in id_list:
        raise HTTPException(status_code=404, detail="当前id不存在")
    return {"id": id}
```

# 3. FastAPI进阶
## 3.1. 中间件
中间件（Middleware）是一个在每次请求进入 FastAPI 应用时都会被执行的函数。它在请求到达实际的路径操作（路由处理函数）之前运行，并且在响应返回给客户端之前再运行一次。

### 3.1.1. 中间件的作用
中间件可以为每个请求添加统一的处理逻辑，常见应用场景包括：
- 记录日志
- 身份认证
- 跨域处理
- 设置响应头
- 性能监控

### 3.1.2. 中间件的定义
在函数的顶部使用装饰器 `@app.middleware("http")` 来定义中间件。

```python
@app.middleware("http")
async def middleware(request, call_next):
    print('中间件开始处理 -- start')
    response = await call_next(request)
    print('中间件处理完成 -- end')
    return response
```

### 3.1.3. 多个中间件的执行顺序
多个中间件的执行顺序是**自下而上**，即后定义的中间件先执行。

## 3.2. 依赖注入
依赖注入（Dependency Injection）是 FastAPI 的核心特性之一，用于共享通用逻辑，减少代码重复。

### 3.2.1. 什么是依赖注入
- **依赖项**：可重用的组件（函数/类），负责提供某种功能或数据
- **注入**：FastAPI 自动帮你调用依赖项，并将结果"注入"到路径操作函数中

### 3.2.2. 依赖注入的优点
- **代码复用**：一次编写，多处使用
- **解耦**：业务逻辑与基础设施代码分离
- **易于测试**：轻松地用模拟依赖替换真实依赖进行测试

### 3.2.3. 依赖注入的应用场景
1. **处理请求参数**：从请求中提取和验证参数（路径参数、查询参数、请求体）
2. **共享数据库连接**：管理数据库会话的创建、使用、关闭
3. **共享业务逻辑**：抽取封装多个路由公用的逻辑代码
4. **安全和认证**：验证用户身份、检查权限和角色要求等

### 3.2.4. 依赖注入的使用
使用依赖注入的步骤：
1. 创建依赖项（函数）
2. 导入 `Depends`
3. 在路径操作函数中声明依赖项

#### 依赖注入实战
> 需求：多个接口都需要分页逻辑，使用依赖注入来共享分页参数处理

##### 代码
main.py
```python
from fastapi import FastAPI, Depends, Query

app = FastAPI()

async def common_parameters(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, le=60)
):
    return {"skip": skip, "limit": limit}

@app.get("/news/news_list")
async def get_news_list(commons = Depends(common_parameters)):
    return {"list": "新闻列表", **commons}

@app.get("/users/user_list")
async def get_user_list(commons = Depends(common_parameters)):
    return {"users": "用户列表", **commons}
```

## 3.3. ORM
ORM（Object-Relational Mapping，对象关系映射）是一种编程技术，用于在面向对象的编程语言和关系型数据库之间建立映射关系。它允许开发者使用面向对象的方式操作数据库，而不需要直接编写 SQL 语句。

### 3.3.1. SQLAlchemy ORM
SQLAlchemy 是 Python 中最流行的 ORM 框架，也是企业应用最多、社区最活跃的 ORM 库。

### 3.3.2. 安装和配置
#### 安装依赖
```bash
uv add sqlalchemy[asyncio] aiomysql
```

#### 创建数据库引擎
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 创建异步引擎
engine = create_async_engine(
    "mysql+aiomysql://user:password@localhost/dbname",
    echo=True
)

# 创建会话工厂
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# 依赖项：获取数据库会话
async def get_database():
    async with async_session() as session:
        yield session
```

### 3.3.3. 定义模型
#### 定义模型基类
```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

#### 定义模型类
```python
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Book(Base):
    __tablename__ = "books"
    
    id = Column(Integer, primary_key=True, index=True)
    bookname = Column(String(100), nullable=False)
    author = Column(String(50))
    price = Column(Float)
```

#### 创建数据表
```python
from sqlalchemy import run_sync

# 创建所有表
async def init_db():
    async with engine.begin() as conn:
        await run_sync(Base.metadata.create_all, conn)
```

### 3.3.4. 数据库操作
#### 查询数据
##### 查询所有数据
```python
from sqlalchemy import select

@app.get("/book/get_books")
async def get_book_list(db: AsyncSession = Depends(get_database)):
    stmt = select(Book)
    result = await db.execute(stmt)
    books = result.scalars().all()
    return {"books": books}
```

##### 查询单条数据
```python
# 方式1：使用 get 方法（根据主键）
book = await db.get(Book, book_id)

# 方式2：使用 select + first
stmt = select(Book).where(Book.id == book_id)
result = await db.execute(stmt)
book = result.scalars().first()

# 方式3：使用 scalar_one_or_none（确保只有一条或没有）
book = result.scalars().one_or_none()
```

##### 分页查询
```python
@app.get("/book/get_books")
async def get_book_list(
    page: int = 1,
    page_size: int = 3,
    db: AsyncSession = Depends(get_database)
):
    skip = (page - 1) * page_size
    stmt = select(Book).offset(skip).limit(page_size)
    result = await db.execute(stmt)
    books = result.scalars().all()
    return {"books": books}
```

#### 新增数据
核心步骤：定义 ORM 对象 → 添加对象到事务：`add(对象)` → `commit` 提交到数据库

```python
from pydantic import BaseModel

class BookBase(BaseModel):
    bookname: str
    author: str
    price: float

@app.post("/book/add_book")
async def add_book(book: BookBase, db: AsyncSession = Depends(get_database)):
    # 创建图书对象（**book.dict() 返回 book 对象的属性字典）
    book_obj = Book(**book.dict())
    db.add(book_obj)
    await db.commit()
    return book
```

#### 更新数据
核心步骤：查询 `get` → 属性重新赋值 → `commit` 提交到数据库

```python
from pydantic import BaseModel
from fastapi import HTTPException

class BookUpdate(BaseModel):
    bookname: str
    author: str
    price: float

@app.put("/book/update_book/{book_id}")
async def update_book(
    book_id: int, 
    data: BookUpdate, 
    db: AsyncSession = Depends(get_database)
):
    # 1. 查询
    book = await db.get(Book, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    # 2. 修改属性(重新赋值)
    book.bookname = data.bookname
    book.author = data.author
    book.price = data.price
    # 3. 提交
    await db.commit()
    return book
```

#### 删除数据
核心步骤：查询 `get` → `delete` 删除 → `commit` 提交到数据库

```python
@app.delete("/book/delete_book/{book_id}")
async def delete_book(
    book_id: int, 
    db: AsyncSession = Depends(get_database)
):
    db_book = await db.get(Book, book_id)
    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    await db.delete(db_book)
    await db.commit()
    return {"message": "Book deleted"}
```

### 3.3.5. ORM 查询总结
#### 核心思路
- `select()` → `db.execute()` → 从 ORM 对象获取数据 → 响应结果
- `db.get(模型类, 主键值)`

#### 从 ORM 对象获取数据的方式
- **获取所有数据**：`scalars().all()`
- **获取单条数据**：
  - `scalars().first()`：提取第一个数据
  - `scalar_one_or_none()`：提取一个或 null
  - `scalar()`：提取标量值（配合聚合查询使用）

### 3.3.6. SQLAlchemy ORM 总结
#### 安装
- `sqlalchemy[asyncio]`
- `aiomysql`（异步数据库驱动）

#### 建表
1. `create_async_engine` 创建异步引擎
2. 定义模型基类（继承 `DeclarativeBase`）和模型类
3. `run_sync(Base.metadata.create_all)` 建表

#### 操作数据
1. 路由处理函数注入数据库会话依赖项 `Depends`
2. 查询数据：`select()`
3. 增加数据：`add()`
4. 更新数据：重新赋值
5. 删除数据：`delete()`