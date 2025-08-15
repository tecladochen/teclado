---

title: 'Cloudreve 个人网盘搭建记录'

publishDate: 2025-08-15 23:26:45

description: '基于阿里云 ECS 和 OSS 搭建内网中转的 Cloudreve 个人网盘'

tags:

 - Cloudreve 搭建

heroImage: { src: './thumbnail.png', color: '#353c67' }

language: '中文'

---

当你有闲置的阿里云 ECS 和 OSS 服务时，可以尝试搭建基于 OSS 的内网中转的 Cloudreve 网盘，喜欢折腾的可以尝试一下，将 Cloudreve 部署在阿里云 ECS 上，然后配置 OSS 内网 Endpoint 使用内网流量 将 OSS 上的文件先下载到 ECS 上，在通过 ECS 的固定带宽下载的自己的设备上。

这套方案上传是直接本地上传到 OSS，流量免费速度很快，下载的速度很大程度取决于 ECS 的公网带宽，这里我 3M 的带宽下载速度稳定在 400KB/s，速度还可以，不搞大文件日常使用够了，主要是基于内网中转不用流出流量费，性价比高。

```mermaid
graph LR
    A[用户 User] --> B[OSS 存储节点]

```

```mermaid
graph RL
    C[OSS 存储节点] --> D[Cloudreve]
    D --> E[用户 User]
```

## 具体步骤
### 准备 docker 环境
阿里云 ECS 安装 Linux 发行版，我这里安装的是 Alibaba Cloud Linux 3，熟悉命令行的话命令行安装好 docker 和 docker compose 环境，不熟悉命令行其实安装宝塔面板更方便，新机器可以直接在镜像市场选择官方的操作系统➕宝塔面板镜像。

我这里使用面板安装 docker，操作非常简单，登陆宝塔面板，点击左侧导航栏下方左边的菜单显示隐藏设置，docker 默认是隐藏的，点击打开，然后在菜单栏进入 docker 页面，点击安装 docker，等待安装完成 docker 和 docker compose 就安装上了，是不是非常容易。

### 安装 Cloudreve