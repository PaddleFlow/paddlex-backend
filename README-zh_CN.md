# paddlex-backend

[English](README.md) | 简体中文

## 概述

paddlex-backend包含了一系列的组件，方便用户在云上落地飞桨框架的生态套件以及深度学习工作流。
目前包含的功能组件有：
- JupyterHub：支持多租户隔离的交互式编程入口
- DataSet: 分布式样本缓存与管理组件
- Training: 支持参数服务器和集合通信两种架构模式的分布式训练组件
- VisualDL: 模型训练日志可视化组件，该组件包含在Training组件中
- ModelHub：模型格式转换、集中存储、版本管理组件
- Serving: 模型推理服务组件，支持蓝绿发版、缩容至零等功能

## 快速上手

#### 前提条件

* Kubernetes >= 1.8
* kubectl

### 安装

如果需要 Kubeflow 提供的多租户隔离功能可以选择，

#### 多租户安装
`hack` 目录下存放着

```bash
sh 
```

#### Standalone安装





