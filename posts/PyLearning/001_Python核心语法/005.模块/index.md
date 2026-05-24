---
title: Python核心语法05-模块
author: Hygen
description: Python模块学习
published: 2026-05-24
---

# 1. Python模块

## 1.1. 导入模块

### 1.1.1. 什么是模块

Python 模块（module）指的是一个 `.py` 文件。

一个模块中可以包含：

1. 变量。
2. 函数。
3. 类。
4. 可执行代码。

简单来说：

> 一个 Python 文件就是一个模块，模块是 Python 程序的基本组织单位。

例如，定义一个 `circle_fun.py` 文件：

```python
# 根据半径计算圆的面积
def circle_area(r):
    pi = 3.14
    area = pi * r * r
    return area

# 根据半径计算圆的周长
def circle_perimeter(r):
    pi = 3.14
    perimeter = 2 * pi * r
    return perimeter
```

这个 `circle_fun.py` 文件就可以看作一个模块。

### 1.1.2. 为什么要使用模块

当代码量比较少时，可以把所有代码写在一个文件中。

但是随着项目变复杂，如果所有代码都写在一个文件中，会出现以下问题：

1. 代码太长，不方便阅读。
2. 功能混在一起，不方便维护。
3. 相同功能无法复用。
4. 多人协作时容易互相影响。

使用模块可以把不同功能拆分到不同文件中，从而让代码结构更加清晰。

模块的主要作用如下：

| 作用 | 说明 |
| :---: | :--- |
| 提高代码复用性 | 一个模块中的函数可以被多个文件重复使用 |
| 降低开发门槛 | 常用功能可以直接导入别人写好的模块 |
| 避免命名冲突 | 不同模块中可以有相同名称的函数或变量 |
| 便于维护管理 | 不同功能拆分到不同文件中，结构更清晰 |

### 1.1.3. 常见内置模块

Python 提供了很多常用的内置模块，例如：

| 模块 | 作用 |
| :---: | :--- |
| `math` | 数学计算 |
| `os` | 操作系统相关操作 |
| `datetime` | 日期时间处理 |
| `re` | 正则表达式 |
| `random` | 随机数相关操作 |
| `sys` | 系统参数相关操作 |
| `time` | 时间访问与处理 |
| `csv` | CSV 文件操作 |

例如，使用 `random` 模块实现随机点名：

```python
import random

names = [
    "王林", "李慕婉",
    "许立国", "韩立",
    "涛哥", "莫厉海",
    "十三", "虎咆",
    "红蝶", "天运子"
]

print(random.choice(names))
```

其中：

- `import random` 表示导入 `random` 模块。
- `random.choice(names)` 表示从列表中随机选择一个元素。

### 1.1.4. 模块导入的基本原则

在使用模块中提供的功能之前，必须先导入模块。

也就是：

> 先导入，再使用。

模块导入语句一般写在 `.py` 文件的开头。

### 1.1.5. import 模块名

第一种方式是直接导入模块。

语法：

```python
import 模块名
```

示例：

```python
import random

num = random.randint(10, 100)
print(num)
```

说明：

- `import random` 表示导入 `random` 模块。
- 调用模块中的功能时，需要使用 `模块名.功能名` 的形式。
- `random.randint(10, 100)` 表示生成 10 到 100 之间的随机整数。

也可以一次导入多个模块：

```python
import random, os

print(random.randint(1, 10))
print(os.getcwd())
```

不过在实际开发中，更推荐分多行导入，代码更清晰：

```python
import random
import os
```

### 1.1.6. import 模块名 as 别名

如果模块名比较长，或者想让代码更简洁，可以给模块起别名。

语法：

```python
import 模块名 as 别名
```

示例：

```python
import random as rd

num = rd.randint(10, 100)
print(num)
```

说明：

- `rd` 是 `random` 模块的别名。
- 使用别名后，调用功能时要写 `rd.randint()`，不能再写 `random.randint()`。

### 1.1.7. from 模块名 import 功能名

如果只想使用模块中的某几个功能，可以只导入指定功能。

语法：

```python
from 模块名 import 功能名
```

示例：

```python
from random import randint, choice

num = randint(10, 100)
print(num)

names = ["张三", "李四", "王五"]
print(choice(names))
```

说明：

- `from random import randint, choice` 表示从 `random` 模块中导入 `randint` 和 `choice`。
- 导入后可以直接使用功能名，不需要再写 `random.`。

### 1.1.8. from 模块名 import 功能名 as 别名

也可以给导入的某个功能起别名。

语法：

```python
from 模块名 import 功能名 as 别名
```

示例：

```python
from random import randint as rint

num = rint(10, 100)
print(num)
```

说明：

- `rint` 是 `randint` 的别名。
- 使用别名后，直接调用 `rint()` 即可。

### 1.1.9. from 模块名 import *

`*` 表示导入模块中的所有功能。

语法：

```python
from 模块名 import *
```

示例：

```python
from random import *

num = randint(10, 100)
print(num)

names = ["张三", "李四", "王五"]
print(choice(names))
```

这种写法虽然简洁，但是不推荐在大型项目中大量使用。

原因是：

1. 不容易看出某个功能来自哪个模块。
2. 容易造成命名冲突。
3. 不利于代码维护。

### 1.1.10. 模块导入方式总结

| 导入形式 | 代码样例 | 调用方式 |
| :--- | :--- | :--- |
| `import 模块名` | `import random` | `random.randint(10, 100)` |
| `import 模块名 as 别名` | `import random as rd` | `rd.randint(10, 100)` |
| `from 模块名 import 功能名` | `from random import randint` | `randint(10, 100)` |
| `from 模块名 import 功能名 as 别名` | `from random import randint as rint` | `rint(10, 100)` |
| `from 模块名 import *` | `from random import *` | `randint(10, 100)` |

常用语法可以概括为：

```python
import 模块名 [as 别名]

from 模块名 import 功能名 [as 别名]

from 模块名 import *
```

## 1.2. 自定义模块

### 1.2.1. 什么是自定义模块

自定义模块就是程序员自己创建的 `.py` 文件。

在实际开发中，当项目比较复杂时，为了让代码结构更清晰、更便于维护和复用，通常会把一个项目拆分为多个模块。

例如，一个完整的项目中可能包含：

```python
config.py
session.py
ai_partner.py
```

其中：

| 文件 | 作用 |
| :---: | :--- |
| `config.py` | 保存项目配置 |
| `session.py` | 处理会话相关功能 |
| `ai_partner.py` | 项目主程序入口 |

注意：

> 每一个 Python 文件都可以作为一个模块，模块名就是文件名。

例如：

```python
circle_fun.py
```

这个模块的模块名就是：

```python
circle_fun
```

导入时不要写 `.py` 后缀：

```python
import circle_fun
```

### 1.2.2. 自定义模块命名规范

自定义模块的文件名建议遵守 Python 标识符命名规范：

1. 使用英文、数字、下划线。
2. 不能以数字开头。
3. 不要使用 Python 关键字。
4. 建议全部小写。
5. 多个单词之间使用下划线连接。

推荐写法：

```python
circle_fun.py
data_query.py
web_service.py
my_config.py
```

不推荐写法：

```python
CircleFun.py
2test.py
my-module.py
class.py
```

### 1.2.3. 自定义模块示例

假设有一个 `circle_fun.py` 文件，内容如下：

```python
# 根据半径计算圆的面积
def circle_area(r):
    pi = 3.14
    area = pi * r * r
    return area

# 根据半径计算圆的周长
def circle_perimeter(r):
    pi = 3.14
    perimeter = 2 * pi * r
    return perimeter
```

然后在 `main.py` 中导入并使用：

```python
import circle_fun

area = circle_fun.circle_area(10)
perimeter = circle_fun.circle_perimeter(10)

print(area)
print(perimeter)
```

输出结果：

```python
314.0
62.800000000000004
```

也可以只导入指定功能：

```python
from circle_fun import circle_area, circle_perimeter

area = circle_area(10)
perimeter = circle_perimeter(10)

print(area)
print(perimeter)
```

### 1.2.4. 项目拆分示例

假设我们要开发一个“AI树洞”项目，如果把所有代码都写在一个文件中，代码会非常混乱。

可以把不同功能拆分到不同模块中。

#### config.py

用于存放基础配置项：

```python
PAGE_TITLE = "AI树洞"
PAGE_ICON = "🤖"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
HELP_URL = "https://www.extremelycoolapp.com/help"
SESSION_LOCATION = "sessions/"
```

#### session.py

用于处理会话相关功能：

```python
import os

def load_sessions():
    sessions = []
    if os.path.exists("sessions"):
        for filename in os.listdir("sessions"):
            if filename.endswith(".json"):
                sessions.append(filename[:-5])
    return sorted(sessions, reverse=True)

def save_session(session_name):
    if session_name:
        # 这里省略保存会话的具体逻辑
        pass
```

#### ai_partner.py

作为主程序文件，导入并使用其他模块中的功能：

```python
from session import load_sessions, save_session

# 新建会话按钮
if st.button("新建会话", icon="✏️", use_container_width=True):
    if st.session_state.current_session:
        save_session(st.session_state.current_session)

    new_session_name = generate_new_session_name()
    st.session_state.current_session = new_session_name
    st.session_state.messages = []

    save_session(new_session_name)
    st.session_state.sessions = load_sessions()
    st.rerun()
```

通过这种拆分方式，可以让每个文件只负责一类功能，项目结构更加清晰。

### 1.2.5. __all__ 变量

`__all__` 是一个模块级别的特殊变量。

它的作用是：

> 控制 `from 模块名 import *` 时会导入哪些功能。

例如，创建一个 `my.py` 文件：

```python
__all__ = ["log_separator1", "log_separator3", "PI"]

PI = 3.1415926
NAME = "黑马☆涛哥"

def log_separator1():
    print("- " * 30)

def log_separator2():
    print("+ " * 30)

def log_separator3():
    print("# " * 30)

def log_separator4():
    print("* " * 30)
```

然后在 `test.py` 中使用：

```python
from my import *

log_separator1()
log_separator3()

print(PI)
```

此时可以使用：

```python
log_separator1
log_separator3
PI
```

但是不能直接使用：

```python
log_separator2
log_separator4
NAME
```

因为它们没有写在 `__all__` 中。

### 1.2.6. __all__ 的注意事项

`__all__` 只控制这种导入方式：

```python
from 模块名 import *
```

它不会影响直接导入指定功能。

例如，即使 `NAME` 没有写入 `__all__`，下面这种写法仍然可以使用：

```python
from my import NAME

print(NAME)
```

所以，`__all__` 的作用并不是彻底隐藏模块中的功能，而是控制 `import *` 时通配导入的范围。

### 1.2.7. __name__ 变量

`__name__` 是 Python 中非常重要的内置变量，表示当前模块的名称。

它有两种常见情况：

| 运行方式 | `__name__` 的值 |
| :---: | :--- |
| 当前模块被直接运行 | `"__main__"` |
| 当前模块被其他文件导入 | 模块名，也就是文件名去掉 `.py` 后缀 |

例如，创建一个 `my.py` 文件：

```python
def test():
    print("这是测试代码")

print(__name__)
```

如果直接运行 `my.py`，输出结果是：

```python
__main__
```

如果在其他文件中导入 `my.py`：

```python
import my
```

此时 `my.py` 中的 `__name__` 输出结果是：

```python
my
```

### 1.2.8. if __name__ == "__main__"

在自定义模块中，经常会看到下面这种写法：

```python
if __name__ == "__main__":
    # 测试代码
    pass
```

它的作用是：

> 只有当前文件被直接运行时，下面的代码才会执行；如果当前文件被其他模块导入，下面的代码不会执行。

示例：

```python
def circle_area(r):
    return 3.14 * r * r

def circle_perimeter(r):
    return 2 * 3.14 * r

if __name__ == "__main__":
    print(circle_area(10))
    print(circle_perimeter(10))
```

这样写的好处是：

1. 可以在模块内部编写测试代码。
2. 测试代码不会在模块被导入时自动执行。
3. 更适合代码复用和项目开发。

### 1.2.9. 自定义模块小结

自定义模块的核心要点如下：

1. 一个 `.py` 文件就是一个模块。
2. 模块名就是文件名，不包含 `.py` 后缀。
3. 模块中可以定义变量、函数、类和可执行代码。
4. 使用模块前需要先导入。
5. 自定义模块可以提高代码复用性和项目可维护性。
6. `__all__` 用于控制 `from 模块名 import *` 时导入哪些功能。
7. `__name__` 用于判断模块是被直接运行，还是被其他模块导入。

## 1.3. 软件包（package）

### 1.3.1. 为什么需要软件包

当项目规模变大时，模块文件会越来越多。

例如，一个项目中可能有很多模块：

```python
data_query.py
data_handle.py
data_model.py
data_analysis.py
web_routes.py
web_auth.py
helpers.py
file.py
format.py
token.py
log.py
web_service.py
```

如果所有模块都放在同一个目录下，就容易造成混乱，不方便管理和维护。

这时就可以使用软件包（package）来组织模块。

### 1.3.2. 什么是包

包的本质是一个文件夹。

这个文件夹中可以包含多个 Python 模块，也就是多个 `.py` 文件。

同时，包文件夹下通常还会包含一个特殊文件：

```python
__init__.py
```

因此，可以简单理解为：

> 包就是用来管理多个模块的文件夹。

例如：

```python
utils/
    __init__.py
    my_var.py
    my_config.py
    my_fun.py
```

其中：

| 文件或目录 | 说明 |
| :---: | :--- |
| `utils/` | 包名 |
| `__init__.py` | 标识当前文件夹是一个包 |
| `my_var.py` | 包中的模块 |
| `my_config.py` | 包中的模块 |
| `my_fun.py` | 包中的模块 |

### 1.3.3. 包的作用

包的主要作用是：

1. 管理多个模块。
2. 对模块进行分类。
3. 让项目结构更加清晰。
4. 提高代码维护性。
5. 避免模块过多造成混乱。

例如，可以按照功能把模块放入不同包中：

```python
project/
    data/
        __init__.py
        data_query.py
        data_handle.py
        data_model.py

    web/
        __init__.py
        web_routes.py
        web_auth.py
        web_service.py

    utils/
        __init__.py
        file.py
        format.py
        log.py
```

这样就能清楚地区分数据处理、网页服务和工具函数等不同功能。

### 1.3.4. __init__.py 的作用

`__init__.py` 是包中的特殊文件。

它的主要作用包括：

1. 标识当前文件夹是一个 Python 包。
2. 可以描述当前包的信息。
3. 可以编写包初始化代码。
4. 可以通过 `__all__` 控制 `from 包名 import *` 时允许导入的模块列表。

例如，在 `utils/__init__.py` 中写入：

```python
__all__ = ["my_fun", "my_var"]
```

表示当执行下面代码时：

```python
from utils import *
```

只会导入：

```python
my_fun
my_var
```

不会导入其他未写入 `__all__` 的模块。

### 1.3.5. 包的导入方式1：import 包名.模块名

语法：

```python
import 包名.模块名
```

示例：

```python
import utils.my_fun

utils.my_fun.log_separator1()
```

说明：

- `utils` 是包名。
- `my_fun` 是包中的模块名。
- 调用功能时，需要写完整路径：`包名.模块名.功能名`。

### 1.3.6. 包的导入方式2：from 包名 import 模块名

语法：

```python
from 包名 import 模块名
```

示例：

```python
from utils import my_fun

my_fun.log_separator1()
```

说明：

- 从 `utils` 包中导入 `my_fun` 模块。
- 调用时可以直接使用 `模块名.功能名`。

### 1.3.7. 包的导入方式3：from 包名 import *

语法：

```python
from 包名 import *
```

示例：

```python
from utils import *

my_fun.log_separator1()
```

注意：

> 使用 `from 包名 import *` 时，需要在包的 `__init__.py` 文件中添加 `__all__`，控制允许导入的模块列表。

例如：

```python
__all__ = ["my_fun", "my_var"]
```

### 1.3.8. 包的导入方式4：from 包名.模块名 import 功能名

语法：

```python
from 包名.模块名 import 功能名
```

示例：

```python
from utils.my_fun import log_separator1

log_separator1()
```

说明：

- 直接从包中的某个模块导入指定功能。
- 使用时可以直接调用功能名。

### 1.3.9. 包的导入方式5：from 包名.模块名 import *

语法：

```python
from 包名.模块名 import *
```

示例：

```python
from utils.my_fun import *

log_separator1()
```

说明：

- 从包中的某个模块导入全部允许导入的功能。
- 如果模块内部定义了 `__all__`，则只导入 `__all__` 中指定的功能。

### 1.3.10. 包的导入方式总结

| 导入形式 | 代码样例 | 调用方式 |
| :--- | :--- | :--- |
| `import 包名.模块名` | `import utils.my_fun` | `utils.my_fun.log_separator1()` |
| `from 包名 import 模块名` | `from utils import my_fun` | `my_fun.log_separator1()` |
| `from 包名 import *` | `from utils import *` | `my_fun.log_separator1()` |
| `from 包名.模块名 import 功能名` | `from utils.my_fun import log_separator1` | `log_separator1()` |
| `from 包名.模块名 import *` | `from utils.my_fun import *` | `log_separator1()` |

### 1.3.11. 包小结

关于包，需要掌握以下内容：

1. 包本质上是一个文件夹。
2. 包中可以存放多个 Python 模块。
3. 包通常包含一个 `__init__.py` 文件。
4. `__init__.py` 可以标识这是一个包，而不是普通文件夹。
5. 当模块较多时，可以用包来分类管理模块。
6. `from 包名 import *` 时，可以通过 `__init__.py` 中的 `__all__` 控制导入范围。

# 2. 本章总结

## 2.1. 模块总结

1. 一个 `.py` 文件就是一个模块。
2. 模块中可以定义变量、函数、类和可执行代码。
3. 使用模块可以提高代码复用性。
4. 使用模块可以让项目结构更加清晰。
5. 使用模块前必须先导入。
6. 模块导入语句一般写在文件开头。

## 2.2. 模块导入方式总结

常见模块导入方式如下：

```python
import 模块名

import 模块名 as 别名

from 模块名 import 功能名

from 模块名 import 功能名 as 别名

from 模块名 import *
```

其中：

1. `import 模块名` 需要通过 `模块名.功能名` 使用。
2. `import 模块名 as 别名` 需要通过 `别名.功能名` 使用。
3. `from 模块名 import 功能名` 可以直接使用功能名。
4. `from 模块名 import *` 会导入模块中允许导入的全部功能，但大型项目中不建议滥用。

## 2.3. 自定义模块总结

1. 自定义模块就是自己创建的 `.py` 文件。
2. 模块名就是文件名，不包含 `.py` 后缀。
3. 自定义模块可以把复杂项目拆分成多个文件。
4. `__all__` 可以控制 `from 模块名 import *` 的导入范围。
5. `__name__` 可以判断模块是直接运行还是被导入。
6. `if __name__ == "__main__"` 常用于编写模块测试代码。

## 2.4. 包总结

1. 包本质上是一个文件夹。
2. 包中可以存放多个模块。
3. 包通常包含 `__init__.py` 文件。
4. 包可以对多个模块进行归类管理。
5. 包的本质也是模块。
6. 当项目模块较多时，使用包可以让项目结构更加清晰。

## 2.5. 包导入方式总结

常见包导入方式如下：

```python
import 包名.模块名

from 包名 import 模块名

from 包名 import *

from 包名.模块名 import 功能名

from 包名.模块名 import *
```

使用建议：

1. 项目较小时，可以直接使用模块。
2. 项目变大后，建议按照功能拆分模块。
3. 模块数量较多时，建议使用包进行分类管理。
4. 尽量少用 `import *`，避免命名冲突。
5. 优先选择结构清晰、来源明确的导入方式。