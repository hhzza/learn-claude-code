# Python 语法基础 —— 来自 learn-claude-code 教学项目的真实模式

> 不需要把 Python 语法全学一遍。这个教学项目反复用的就是这十几个模式，吃透就能读懂所有章节的代码。

---

## 一、dataclass — 数据类（不用手写 `__init__`）

```python
from dataclasses import dataclass, asdict, field

@dataclass
class Task:
    id: str                          # 必填字段
    subject: str                     # 必填字段
    status: str = "pending"          # 有默认值，可不传
    blockedBy: list[str] | None = None  # 可选类型（Python 3.10+）
    created_at: float = field(default_factory=time.time)  # 动态默认值

# 用法
task = Task(id="t1", subject="写测试")   # 自动生成 __init__
data = asdict(task)                       # 转成字典，方便 json.dumps
json.dumps(data, indent=2)                # 写文件
```

**原则**：结构化数据全部用 dataclass，不手写类。

---

## 二、字典/列表推导式（最常用的数据转换）

```python
# 字典：工具名 → 函数 的映射
handler = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
}.get(block.name)           # .get() 找不到返回 None，不会报 KeyError

# 列表推导
lines = [f"  {t.id}: {t.subject}" for t in tasks]

# 带条件的列表推导
unblocked = [t.subject for t in list_tasks()
             if t.status == "pending" and can_start(t.id)]

# 对文件列表做推导
tasks = [Task(**json.loads(p.read_text()))
         for p in sorted(TASKS_DIR.glob("task_*.json"))]
```

---

## 三、`with` 语句（资源管理，自动关闭/释放）

```python
# 写文件：不用手动 close()
with open(inbox, "a") as f:
    f.write(json.dumps(msg) + "\n")

# 线程锁：不用手动 release()
with background_lock:
    background_tasks[bg_id] = {"status": "running"}
```

**原则**：`open()` 或 `threading.Lock()` 后面永远跟 `with`。

---

## 四、pathlib — 路径操作（替代 `os.path`）

```python
from pathlib import Path

WORKDIR = Path.cwd()                    # 当前目录（不是 os.getcwd()）
MEMORY_DIR = WORKDIR / ".memory"        # 用 / 拼接路径
MEMORY_DIR.mkdir(exist_ok=True)         # 创建目录，存在不报错

# 文件操作
path = MAILBOX_DIR / f"{agent}.jsonl"   # 拼接文件名
path.exists()                           # 判断存在
path.read_text()                        # 读全部文本（不是 open().read()）
path.write_text(data)                   # 写全部文本（不是 open().write()）
path.read_text().splitlines()           # 读成行列表
path.unlink()                           # 删除文件

# 其他
path.parent                             # 父目录
path.is_relative_to(WORKDIR / ".memory")  # 安全检查：路径不越界
sorted(TASKS_DIR.glob("task_*.json"))   # 按文件名模式搜索（不是 glob 模块）
```

---

## 五、subprocess — 跑 shell 命令

```python
import subprocess

r = subprocess.run(
    command,
    shell=True,              # 用 shell 解析命令
    cwd=WORKDIR,             # 工作目录
    capture_output=True,     # 捕获 stdout + stderr
    text=True,               # 返回字符串，不是 bytes
    timeout=120              # 超时秒数
)
out = (r.stdout + r.stderr).strip()      # 合并输出
out = out[:50000] if out else "(no output)"  # 截断防爆
```

---

## 六、threading — 多线程

```python
import threading

# 创建锁
lock = threading.Lock()

# 创建后台线程
def worker():
    result = do_something()
    with lock:
        shared[bg_id] = result

t = threading.Thread(target=worker, daemon=True)
t.start()
# daemon=True → 主线程退出时自动回收，不会成僵尸线程

# 在线程里操作共享字典，加锁
with background_lock:
    background_tasks[bg_id] = {"status": "running"}
```

**模式**：内部函数 `def run():` + `target` + `daemon=True` + 锁保护共享数据。

---

## 七、json 读写

```python
import json

# 普通 JSON：整个文件一个对象
path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
data = json.loads(path.read_text())

# JSONL：每行一个 JSON 对象（文件收件箱用这个）
with open(inbox, "a") as f:
    f.write(json.dumps(msg) + "\n")      # 追加一行

# 读 JSONL
msgs = [json.loads(line) for line in path.read_text().splitlines()
        if line.strip()]

# ** 解包：把字典展开成关键字参数
task = Task(**json.loads(path.read_text()))  # 等价于 Task(id=..., subject=...)
```

---

## 八、f-string — 字符串格式化

```python
# 插变量
f"Task {task.id} is {task.status}"

# 截断
f"From {m['from']}: {m['content'][:200]}"

# 颜色（终端）
f"\033[36m[claim] {task.subject}\033[0m"
# \033[...m = ANSI 颜色码，\033[0m = 重置
```

---

## 九、`global` — 修改模块级变量

```python
_bg_counter = 0  # 模块级变量

def start_background_task():
    global _bg_counter    # 要赋值必须声明 global
    _bg_counter += 1      # 否则 Python 会当成局部变量，报 UnboundLocalError
```

**规则**：读不需要 `global`，只有**赋值**（`=`、`+=`、`-=`）才需要。

---

## 十、lambda — 匿名函数

```python
# 需要传一个简单函数但不想写 def
"send_message": lambda to, content: (BUS.send(name, to, content), "Sent")[1],

# 等价于：
def handler(to, content):
    BUS.send(name, to, content)   # 副作用
    return "Sent"                 # 返回值

# `(expr1, expr2)[1]` 模式：
#   元组里的两个表达式都会执行
#   [1] 取第二个元素作为返回值
#   因为 lambda 只能写单行，所以用元组包住两句
```

---

## 十一、命名约定（项目全程一致）

| 约定 | 例子 | 含义 |
|------|------|------|
| `_prefix` | `_bg_counter`, `_last_fired`, `_task_path()` | 模块内部用的，外面别碰 |
| `UPPER_CASE` | `WORKDIR`, `MAILBOX_DIR`, `TOOLS` | 启动时赋值一次，当常量用 |
| `snake_case` | `spawn_teammate_thread`, `read_inbox` | 普通函数和变量 |
| `run_xxx` | `run_bash`, `run_read`, `run_create_task` | tool handler 函数 |
| `list_xxx` | `list_tasks`, `list_crons` | 返回列表的函数 |
| `get_xxx` | `get_task`, `get_system_prompt` | 返回单个对象的函数 |

---

## 十二、Agent Loop 骨架（背下来，每章都一样）

```python
def agent_loop(messages: list, context: dict):
    system = get_system_prompt(context)
    while True:
        # 1. 注入定时任务（可选）
        fired = consume_cron_queue()

        # 2. 调 LLM
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
            tools=TOOLS, max_tokens=8000)

        # 3. 加到历史
        messages.append({"role": "assistant", "content": response.content})

        # 4. 不需要工具 → 结束
        if response.stop_reason != "tool_use":
            return

        # 5. 执行工具
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_MAP.get(block.name)
                output = handler(**block.input)  # ** 解包参数
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # 6. 工具结果加入历史，继续循环
        messages.append({"role": "user", "content": results})
```

---

## 十三、文件读写的三种模式

```python
# 读 → 改 → 写回（任务系统用这个模式）
path.write_text(json.dumps(data))

# 追加（收件箱用，因为多个线程可能同时写入）
with open(inbox, "a") as f:
    f.write(json.dumps(msg) + "\n")

# 读 + 删除（消费式收件箱）
msgs = json.loads(path.read_text().splitlines())
path.unlink()  # 读完即删
```

---

## 十四、每个章节的代码模板

按这个顺序写，每个新模块就只是在某个环节加东西：

```
1. import 老三样（os, json, time, threading, pathlib, anthropic, dotenv）
2. WORKDIR + client + MODEL 常量
3. @dataclass 数据结构
4. def xxx_path(id) → Path          # 辅助函数
5. def create_xxx(...) → Xxx        # 创建 + 写文件
6. def list_xxx() → list[Xxx]       # 扫文件 + 解析
7. def run_xxx(...) → str           # tool handler
8. TOOLS 列表定义
9. agent_loop()                     # LLM 主循环
10. if __name__ == "__main__":      # 入口 + 用户输入循环
```

---

## 十五、Python 3.10+ 类型写法速查

```python
# 基本类型
name: str
count: int
active: bool
items: list[str]              # 列表，不是 List[str]（旧写法）
mapping: dict[str, int]       # 字典
maybe: str | None             # 可为 None，不是 Optional[str]（旧写法）
func: callable                # 函数类型（很少用）

# dataclass 的 field
from dataclasses import field
created_at: float = field(default_factory=time.time)  # 每次创建取新值
items: list[str] = field(default_factory=list)         # 默认空列表
```
