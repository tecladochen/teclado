---

title: 'uv 入门使用指南'

publishDate: 2025-07-23 16:45:00

description: 'An extremely fast Python package and project manager, written in Rust.'

tags:

 - uv

heroImage: { src: '../../thumbnails/bluesky.jpg', color: '#026695' }

language: '中文'

---

Python 生态系统中的包管理一直是开发者关注的焦点。从最初的 pip 到后来的 Poetry、PDM 等工具，每一次演进都试图解决前代工具的痛点。而今天我们要介绍的 UV，作为由 Ruff 团队（Astral）开发的下一代 Python 包管理工具，正以其惊人的速度和创新的设计理念，重新定义 Python 包管理的标准。

## UV 介绍

### 什么是 UV

UV 是一个极速的 Python 包管理器和解析器，由 Rust 编写，专注于提供卓越的性能和用户体验。其名称 "UV" 源自 "μv"（微伏），暗示着其轻量级和高效的特性。作为 pip 的直接替代品，UV 提供了兼容的命令行接口，同时在速度上实现了数量级的提升。

### UV 的核心优势

与传统的 Python 包管理工具相比，UV 具有以下显著优势：

- **🚀 一体化工具**：一个工具替代 pip、pip-tools、pipx、poetry、pyenv、twine、virtualenv 等多种工具
- **⚡️ 极致速度**：比 pip 快 10-100 倍
- **🗂️ 全面项目管理**：提供通用锁文件的综合项目管理功能
- **❇️ 脚本运行**：支持带有内联依赖元数据的脚本运行
- **🐍 Python 版本管理**：安装和管理不同的 Python 版本
- **🛠️ 工具安装**：运行和安装以 Python 包形式发布的工具
- **🔩 兼容pip接口**：提供熟悉的命令行界面，同时大幅提升性能
- **🏢 工作区支持**：支持 Cargo 风格的工作区，适用于大规模项目
- **💾 高效磁盘空间利用**：通过全局缓存实现依赖去重
- **⏬ 简易安装**：无需 Rust 或 Python 环境，可通过 curl 或 pip 直接安装
- **🖥️ 多平台支持**：支持 macOS、Linux 和 Windows 系统

随着 Python 项目规模和复杂度的增长，传统工具在性能和用户体验方面的局限性日益凸显。UV 的出现，为开发者提供了一个更快、更可靠的选择，特别适合大型项目和 CI/CD 环境中的应用。

## 安装

UV 的安装非常简单，支持多种操作系统和安装方式。以下是几种常见的安装方法：

**使用官方安装脚本（推荐）**

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**使用包管理器**

```bash
# macOS (Homebrew)
brew install uv
```

安装完成后，可以通过以下命令验证安装是否成功：

```bash
uv --version
```

安装正确会得到版本号：

```bash
uv 0.8.0 (0b2357294 2025-07-17)
```
## python 版本管理

如果系统中已安装 Python，uv 会**自动检测并使用**，无需额外配置。不过，uv 也能够安装和管理 Python 版本，uv 会根据需要自动安装缺失的 Python 版本，因此你无需预先安装 Python 即可上手。

安装最新版本的 Python：

```bash
uv python install
```

Python 官方并未发布可分发的二进制文件，uv 使用的是 Astral 的 [python-build-standalone](https://github.com/astral-sh/python-build-standalone) 项目提供的发行版。uv 安装的 Python 不会全局可用（即无法通过 python 命令调用）需要使用 uv run 或创建并激活虚拟环境来使用下载的 Python。

Python 版本由 Python 解释器（即 python 可执行文件）、标准库和其他支持文件组成，由于系统中通常已安装 Python，uv 支持**发现** Python 版本。不过，uv 也支持 **自行安装** Python 版本。为区分这两种 Python 安装类型，uv 将其安装的 Python 版本称为*托管* Python 安装，而将所有其他 Python 安装称为*系统* Python 安装。

uv 不会区分操作系统安装的 Python 版本与其他工具安装和管理的 Python 版本。例如，如果使用 pyenv 管理 Python 安装，在 uv 中它仍会被视为*系统* Python 版本。

安装特定版本的 Python：

```bash
uv python install 3.12
```

安装多个 Python 版本：

```bash
uv python install 3.11 3.12
```

安装其他 Python 实现，例如 PyPy

```bash
uv python install pypy@3.10
```

要重新安装由 uv 管理的 Python 版本，使用参数 reinstall，例如：

```bash
uv python install 3.12 --reinstall
```

要查看可用和已安装的 Python 版本：

```bash
uv python list
```

卸载安装的 Python 版本

```bash
uv python uninstall 3.12
```

## 脚本运行

这里的脚本指的是独立执行的 python 文件，例如使用 python example.py 来执行 example.py 这个脚本文件，使用 uv 执行脚本可以在无需手动管理环境的情况下管理脚本的依赖项。也就是说传统的使用 python 解释器运行脚本文件的方式下，当脚本需要第三方软件包依赖时，需要在全局环境中安装依赖，意味脚本依赖管理混乱，使用 uv 可以在不用手动管理环境的情况下为每个脚本管理依赖和创建临时虚拟环境。

当运行无依赖或者只依赖标准库中的某些模块的脚本时，直接使用 uv run 来执行脚本。需要注意的是，pyproject.toml 文件用来确定项目根目录，当 uv run 时如果目录中存在 pyproject.toml 文件 uv 会安装文件中的依赖，当脚本在项目中但是不依赖项目时，使用 --no-project 选项来跳过检查项目文件。

```bash
# 注意：`--no-project` 标志必须在脚本名称之前提供。
uv run --no-project example.py
```

当你的脚本需要其他软件包时，必须将它们安装到脚本运行的环境中，uv 倾向于按需创建这些环境，而不是使用手动管理依赖项的长期虚拟环境。按需创建环境意味着当脚本执行时由 uv 来创建所需的环境，区别于项目的手动管理环境。对于脚本所需的依赖项，通常使用**项目**或**内联元数据**来声明依赖项，但 uv 也支持每次调用时请求依赖项。

例如，以下脚本需要 rich 这个包：

```python
# example.py
import time
from rich.progress import track

for i in track(range(20), description="For example:"):
    time.sleep(0.05)
```

如果在未指定依赖项的情况下执行，此脚本将失败：

```bash
uv run --no-project example.py	
```

使用 --with 选项请求依赖项：

```bash
uv run --with rich example.py
```

Python 最近为[内联脚本元数据](https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata)添加了一种标准格式，它允许选择 Python 版本并定义依赖项。使用 uv init --script 来初始化带有内联元数据的脚本：

```bash
uv init --script example.py --python 3.13
```

内联元数据格式允许在脚本本身中声明脚本的依赖项，uv 支持添加和更新内联脚本元数据，使用 uv add --script 来声明脚本的依赖项：

```bash
uv add --script example.py 'requests<3' 'rich'
```

这将在脚本顶部添加一个 script 部分，使用 TOML 声明依赖项：

```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])
```

使用内联脚本元数据时，即使 uv run 在项目中使用，项目的依赖项也将被忽略，无需使用 --no-project 选项，uv 将自动创建一个包含运行脚本所需依赖项的环境来执行脚本。

## 使用工具

### 运行工具

这里的工具指的是提供**命令行交互**的软件包，uv 包含一个用于与工具交互的专用接口，可以使用 uv tool run 无需安装即可调用工具，在这种情况下，其依赖项将安装在与当前项目隔离的临时虚拟环境中。官方提供 uvx 作为 uv tool run 命令的别名，这两个命令完全等效，官方推荐使用 uvx。

uvx 命令可在不安装工具的情况下调用它，可在工具名称后提供参数：

```bash
uvx ruff
# 等效于
uv tool run ruff
```

```bash
uvx pycowsay hello from uv
  -------------
< hello from uv >
  -------------
   \   ^__^
    \  (oo)\_______
       (__)\       )\/\
           ||----w |
           ||     ||
```

当调用 uvx ruff 时，uv 会安装提供 ruff 命令的 ruff 软件包。然而，有时软件包名和命令名会有所不同，可使用 --from 选项从特定软件包调用命令，例如由 httpie 提供的 http 命令：

```bash
uvx --from httpie http
```

--from 还可以指定请求特定版本和不同源的软件包：

```bash
uvx --from 'ruff==0.3.0' ruff check
uvx --from git+https://github.com/httpie/cli httpie
```

### 安装工具

如果经常使用某个工具，就可以将其安装到持久化环境并添加到环境 PATH 中，这样就无需反复调用 uvx 运行工具了。与工具相关的命令都以 uv tool 开始，因为 run 命令常用，所以把 uvx 作为 uv tool run 的别名使用。

安装 ruff 工具：

```bash
uv tool install ruff
```

安装工具后，其可执行文件会被放置在 PATH 中的 bin 目录下，这样就可以不通过 uv 来运行该工具。如果它不在 PATH 中，会显示一条警告，此时可以使用 uv tool update-shell 将其添加到 PATH 中。

安装 ruff 后，它应该就可以使用了：

```bash
ruff --version
```

使用 uv tool install 安装工具时，将在 uv 工具目录中创建一个虚拟环境，除非卸载该工具，否则该环境不会被删除，如果手动删除该环境，工具将无法运行，查看工具目录位置：

```bash
uv tool dir
```

需要注意，安装工具并不能使其模块在当前环境中可用，uv 隔离管理了工具、脚本和项目环境，来减少相互影响和冲突。

与 uvx 不同，uv tool install 操作的是一个 *包*，并且会安装该工具提供的所有可执行文件，所以无需 --from 即可指定包版本和来源：

```bash
uv tool install 'httpie>0.1.0'
uv tool install git+https://github.com/httpie/cli
```

## 项目管理

### 创建新项目

你可以使用 uv init 命令创建一个新的 Python 项目：

```bash
uv init hello-world
cd hello-world
```

或者，你也可以在当前工作目录中初始化一个项目：

```bash
mkdir hello-world
cd hello-world
uv init
```

### 项目结构

一个项目由几个重要部分组成，它们协同工作，使 uv 能够管理你的项目，除了 uv init 创建的文件外，在你首次运行项目命令（如 uv run、uv sync 或 uv lock）时，uv 还会在项目根目录中创建一个虚拟环境和 uv.lock 文件。

完整的文件列表如下：

```bash
.
├── .venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

pyproject.toml 包含项目的元数据，这个文件用来记录依赖项，以及项目的详细信息，如项目描述或许可证，你可以手动编辑此文件，一般来说是使用 uv add 和 uv remove 等命令从终端管理项目。

```toml
[project]
name = "hello-world"
version = "0.1.0"
description = "在此处添加项目描述"
readme = "README.md"
dependencies = []
```

.python-version 文件包含项目的默认 Python 版本，此文件告诉 uv 在创建项目的虚拟环境时应使用哪个 Python 版本。

.venv 文件夹包含项目的虚拟环境，这是一个与系统其他部分隔离的 Python 环境，uv 将在此处安装项目的依赖项。

uv.lock 是一个跨平台的锁定文件，其中包含有关项目依赖项的确切信息，与用于指定项目大致要求的 pyproject.toml 不同，锁定文件包含安装在项目环境中的确切解析版本，此文件应提交到版本控制系统，以便在不同机器上实现一致且可重现的安装，uv.lock 是一个人类可读的 TOML 文件，但由 uv 管理，不应手动编辑。

### 管理依赖

你可以使用 uv add 命令将依赖项添加到 pyproject.toml 中，这也会更新锁定文件和项目环境：

```bash
uv add requests
```

要移除一个包，可以使用 uv remove：

```bash
uv remove requests
```

要升级一个包，可以使用带 --upgrade-package 标志的 uv lock 命令：

```bash
uv lock --upgrade-package requests
```

--upgrade-package 选项会尝试将指定的包更新到最新的兼容版本，同时保持锁定文件的其余部分不变。

### 运行命令

uv run 可用于在项目环境中运行任意脚本或命令，在每次调用 uv run 之前，uv 会验证锁定文件是否与 pyproject.toml 保持同步，并且环境是否与锁定文件保持同步，从而无需手动干预即可使项目保持最新状态。

```bash
uv add flask
uv run -- flask run -p 3000
```

> 注意这里 -- 是 Unix/Linux 命令行通用约定，cmd1 -- cmd2 args 显式声明 cmd2 args 作为完整命令执行，在这里 uv run 激活虚拟环境，然后 flask run -p 3000 单独执行。

或者，你可以使用 uv sync 手动更新环境，然后在执行命令前激活环境。

```bash
uv sync
source .venv/bin/activate
flask run -p 3000
```

### 构建项目

uv build 可用于为你的项目构建源发行版和二进制发行版（wheel），默认情况下，uv build 将在当前目录中构建项目，并将构建产物放置在 dist 子目录中：

```bash
uv build
ls dist/
```

### 发布包

uv 支持通过 uv build 将 Python 包构建为源码和二进制发行版，并通过 uv publish 将它们上传到注册中心。

```bash
uv publish
```

## 结语

文档看到现在基本了解的 uv 的用法和原理，因为大多数编程语言都有自己的包管理器，所以理解起来还不算太难，更多操作细节需要在实践中去摸索。

