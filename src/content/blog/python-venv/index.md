---

title: venv 使用和原理

publishDate: 2025-07-16 10:45:00

description: '关于 python 内置 venv 模块的使用和原理'

tags:

 - venv

heroImage: { src: '../../thumbnails/nightmoon.jpg', color: '#152b33' }

language: '中文'

--- 

在学习 uv 的过程中对 uv venv 虚拟环境的使用一直有点困惑，研究后发现 uv venv 底层是基于 python 内置的 venv 模块实现，但是我对 venv 也不了解，而且之前一直把虚拟环境和 Conda 的独立环境混在一起所以这篇是记录一下 python venv 模块原理和使用。

## 为什么需要虚拟环境

虚拟环境要做的事是**隔离项目依赖**，想象一下这样一个场景，你有两个 python 项目使用到了 Flask 2.0.1，如果其中一个项目需要升级 Flask 版本到 3.0，那么问题来了，如果你是在系统全局 python 环境中安装的 Flask，当你把 Flask 从 2.0.1 升级到 3.0 后，就会破坏另一个项目的 Flask 依赖（因为它被锁定使用 Flask 2.0.1），如果另外再安装一个 Flask 2.0.1 来供原来的项目使用，就会让所有项目的包都混在一起管理。

Python 3.3+ 内置了 venv 模块（替代早期的 virtualenv），来为每个项目创建隔离的 Python 运行环境。

## 创建和使用虚拟环境

### **创建环境**

```bash
# 创建名为 .venv 的虚拟环境（推荐使用 .venv 作为默认名）
python -m venv .venv
```

### 激活环境

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 激活后提示符变化示例：
(.venv) user@machine:~$
```

### 使用环境

```bash
# 安装包（隔离在虚拟环境中）
(.venv) pip install django==4.2

# 运行代码
(.venv) python my_app.py

# 导出依赖
(.venv) pip freeze > requirements.txt
```

### 退出环境

```bash
deactivate
```

## 底层原理

当你使用 python -m venv .venv 创建一个的虚拟环境后，你会看到类似这样的目录结构（在 Unix/macOS 为 bin，Windows 中 Scripts 代替 bin）：

```markdown
.venv/
├── bin/               # Unix/macOS 执行文件
│   ├── python         -> /usr/bin/python3.10 (符号链接)
│   ├── pip            # 专属 pip 脚本
│   └── activate       # 环境激活脚本
├── Scripts/           # Windows 执行文件
│   ├── python.exe     # 解释器链接
│   ├── pip.exe
│   └── activate.bat
├── lib/
│   └── python3.10/
│       └── site-packages/  # 核心！所有安装的包存放于此
└── pyvenv.cfg          # 配置文件
```

最重要的部分是 site-packages 目录，当激活虚拟环境后，后续所有安装的包都安装到这个目录下，系统全局的 site-packages 目录将不再访问。

venv 的本质是将路径重定向到指定的环境目录，在激活环境时主要做了两件事：

1. **修改 PATH 环境变量：**

   将虚拟环境目录下的 bin (或 Scripts) 目录添加到你的系统 PATH 变量的**最前面**，这样，当你直接在命令行输入 python 或 pip 时，系统会优先找到并使用虚拟环境里的版本 ，而不是系统全局的那个。

2. **设置 VIRTUAL_ENV 环境变量：**

   这个变量指向你的虚拟环境目录，很多工具会检查这个变量，知道当前在哪个虚拟环境中工作。

.venv/bin/python 通常只是一个指向你创建环境时指定的那个 Python 解释器（比如 /usr/bin/python3.10）的**符号链接**（或 Windows 上的快捷方式/小副本），它**没有**在虚拟环境内部安装一个完全独立的 Python 副本（不像 Conda）。
