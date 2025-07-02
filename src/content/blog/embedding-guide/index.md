---

title: 理解 Embedding（嵌入）

publishDate: 2025-07-03 02:55:00

description: '一种将离散数据映射为连续变量并且能捕获潜在关系的向量化技术'

tags:

 - NLP

heroImage: { src: './thumbnail.png', color: '#dfbe8e' }

language: '中文'

draft: true

---

## 什么是 Embedding

在大模型中，”embedding“指的是将某种类型的输入数据（如文本、图像、声音等）转换成一个稠密的数值向量的过程。这些向量通常包含较多纬度，每一个纬度代表输入数据的某种抽象特征和属性。embedding 的目的是将实际的输入转换为计算机能够更有效处理的向量格式。

image：文本数据 =》 embedding=》向量数据

在自然语言处理（NLP）中通常使用embedding模型将文本数据转换成embedding向量，这些embedding向量捕获了文本的语义特征，在embedding向量空间中，语义相近的实体在向量空间中距离更近，语义不相近的实体在向量空间中距离更远。

image：embeding向量空间示意图

由于embedding模型有这种精确理解语义的能力，通过对于不同句子段落的语义相似度的计算，embedding模型可以应用在检索增强生成（RAG）、推荐系统这些领域。后面我都以构建RAG为例来学习Embedding模型。

注意：在后面学习transformer架构是会发现模型架构中会一个embedding层，这里的embedding层和embedding模型存在差异，核心区别是embedding层是服务于生成任务的内部零件，而embedding模型是专注于理解的最终的完整产品。所以他们的目标、训练方式和优化方向完全不同。

## Embedding 的底层

当前主流的上下文嵌入模型（例如BAAI/bge-large-zh-v1.5）底层基于 bert 模型，而 bert 模型是基于 Transformer Encoder-Only 架构。