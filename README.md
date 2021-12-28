# 云上飞桨产品介绍

## 介绍

飞桨（PaddlePaddle）是全面开源开放、技术领先、功能完备的产业级深度学习平台，目前已广泛应用于工业、农业、服务业等。
在众多飞桨用户中，一些客户还具有业务上云的需求，这就要求飞桨生态能提供完善且易用的云上组件，方便用户在云上落地飞桨深度学习工作流。
虽然目前云上飞桨已有样本缓存、模型训练、模型服务等云原生组件，但缺少将各组件串起来共同完成飞桨深度学习工作流的工具。
此外，由于大部分飞桨套件用户是各领域的模型开发人员，可能对 Kubernetes 相关的知识并不熟悉。
因此，基于各飞桨生态套件构建该领域的常见深度学习工作流是十分有必要的，此外还需要为用户提供简单易用的编程接口，方便用户构建其自定义的工作流。
基于以上原因，我们构建了云上飞桨产品旨在协助飞桨用户快速地完成深度学习工作流上云的需求。

## 概述

云上飞桨产品包含了一系列的组件，方便飞桨框架和生态套件用户在云上训练和部署模型。

<div align="center">
  <img src="./docs/images/paddle-cloud-arch.png" style="zoom:50%;" />
</div>

上图展现了云上飞桨产品的整体架构，其以云原生的方式构建在不可变基础设施 Kubernetes 之上。
图中在 Kubernetes 之上一共有4层，分别是 Web UI 层、编程接口（API）层、领域应用工作流 Operator 层、云原生依赖组件层 ， 其中浅蓝色部分为飞桨的生态套件，橘黄色部分为外部依赖组件。
各层组件功能如下。

**面向模型开发人员的 Web UI 层：**

**JupyterHub** 支持多用户隔离的交互式编程入口，可以供算法团队多人使用。空闲时资源能够自动释放。
**Pipeline UI** 让用户可以很方便的使用 Python SDK 来构建飞桨深度学习工作流，并在 Pipeline UI 界面上观察工作流的状态。
**VisualDL** 是飞桨可视化分析工具，方便用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。

**面向工程开发人员的编程接口层：**

**Dataset** 定义了样本数据集存储、缓存、管理、预处理等操作接口。
**Training** 封装了基于飞桨框架的模型训练接口，支持参数服务器和集合通信两种分布式架构模式。
**ModelHub** 定义了模型格式转换、集中存储、版本管理等操作接口。
**Serving** 封装了模型推理服务相关接口，支持自动扩缩容、蓝绿发版、缩容至零等功能。

**封装了领域应用工作流的自运维 Operator 层：**

我们将为每个飞桨生态套件提供了该领域常用的深度学习工作流，用户可以开箱即用。
同时，也支持用户通过 DSL Python SDK 灵活地自定义组件和工作流。

**PaddleOCR Operator** 封装了 OCR 领域的常见深度学习工作流，如文本识别、版面分析、表格识别等工作流。
**PaddleClas Operator** 封装了图像识别领域的常见深度学习工作流，如图像分类、图像识别等工作流
**PaddleNLP Operator** 封装了 NLP 领域的常见深度学习工作流，如情感分析、阅读理解、文本翻译、对话系统等工作流。
**PaddleRec Operator** 封装了推荐系统领域的深度学习工作流，包括内容理解、匹配、召回、精排等工作流。

**云上飞桨产品依赖的云原生组件：**

**Argo** 是面向云原生的工作流调度引擎，paddlex-backend 基于此编排作业执行流程。
**Istio** 为云上飞桨产品的 Web UI 和 模型推理服务提供了集群外部可访问的统一的网关。
**Knative** 为模型推理服务提供了自动扩缩容、蓝绿发版、缩容至零等功能。
**JuiceFS** 提供了支持 POXIS 接口的 CSI 插件，用于存储样本数据、模型训练任务的临时文件。

## 安装

### 前提条件

- Kubernetes >= 1.8
- kubectl
- Redis（可以买云服务，或者本地安装单机版）

### 全功能一键脚本安装（可选）

如您需要完备的多租户隔离的功能，可以采用一键脚本安装的方式，在运行脚本前，需要先安装依赖的 [kustomize](https://kustomize.io/) 工具。

```bash
# 下载项目代码到本地
$ git clone https://github.com/PaddleFlow/paddlex-backend
$ cd hack && sh install.sh
```

以上命令会在 Kubernetes 集群中安装所有的依赖组件，需要等待一段时间。然后您可以通过以下命令来确认所有依赖的组件是否安装成功。

```bash
$ kubectl get pods -n cert-manager
$ kubectl get pods -n istio-system
$ kubectl get pods -n auth
$ kubectl get pods -n knative-eventing
$ kubectl get pods -n knative-serving
$ kubectl get pods -n kubeflow
$ kubectl get pods -n kubeflow-user-example-com
```

通过 Port-Forward 暴露 UI 界面的端口，从而访问 UI 界面。

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

打开浏览器访问 `http://localhost:8080`，默认用户名是 `user@example.com`，默认密码是 `12341234`

### standalone模式安装 (可选)

如果您并不要多租户隔离功能，可以选着更加轻量级的 standalone 模式安装。

```bash
$ git clone https://github.com/PaddleFlow/paddlex-backend
$ cd hack && sh install-standalone.sh
```

以上命令会安装 Istio/knative/kubeflow pipeline/paddle-operator/JupyterHub 等组件，您可以通过以下命令查看各组件部署的情况。

```bash
$ kubectl get pods -n kubeflow
$ kubectl get pods -n istio
$ kubectl get pods -n knative
```

### 创建 StorageClass 和 Secret

由于使用的 JuiceFS CSI 插件需要使用 Redis 作为元数据存储，您可以选着使用云服务商提供的 Redis 服务或者是自己安装。
然后，将 Redis 服务的连接URI进行Base64编码，URI的格式为 `redis://[username]:[passward][@]IP:PORT/DATABASE`。
如 `redis://default:123456@127.0.0.1:6379/0`，使用命令 `echo "redis://default:123456@127.0.0.1:6379/0" | base64 ` 获取 Base64 编码。
然后将 Base64 编码后的字符串填入到 hack/prerequisites 目录下文件 data-center.yaml 和 task-center 的字段 metaurl 上。
最后执行如下命令，创建 StorageClass 和 Secret。

```bash
$ kubectl apply -f ./prerequisites/data-center.yaml
$ kubectl apply -f ./prerequisites/task-center.yaml
```

## 示例

更多示例可以参考 [PP-OCR Pipeline](./operator/README.md)。

## API

**Dataset API 相关的字段：**

|字段 |类型 |说明 |
|---|---|---|
|name|String|数据集名称，如：imagenet|
|namespace|String|命名空间，默认为kubeflow|
|action|String|对自定义资源的操作，apply/create/patch/delete之一|
|partitions|Integer|样本数据集缓存分区数，一个分区表示一个节点，默认为1|
|source_uri|String|样本数据的存储地址，如 bos://paddleflow/imagenet/|
|source_secret|String|样本数据存储后端的访问秘钥|

**Training API 相关的字段：**

|字段 |类型 |说明 |
|---|---|---|
|name|String|模型名称，如: pp-ocr|
|namespace|String|命名空间，默认为kubeflow|
|action|String|对自定义资源的操作，apply/create/patch/delete之一|
|project|String|飞桨生态套件项目名，如 PaddleOCR/PaddleClas等|
|image|String|包含飞桨生态套件代码的模型训练镜像|
|config_path|String|模型配置文件的在套件项目中的相对路径|
|dataset|String|模型训练任务用到的样本数据集|
|pvc_name|String|工作流共享盘的PVC名称|
|worker_replicas|Integer|分布式训练 Worker 节点的并行度|
|config_changes|String|模型配置文件修改内容|
|pretrain_model|String|预训练模型存储路径|
|train_label|String|训练集的样本标签文件名|
|test_label|String|测试集的样本标签文件名|
|ps_replicas|Integer|分布式训练参数服务器并行度|
|gpu_per_node|Integer|每个 Worker 节点所需的GPU个数|
|use_visualdl|Boolean|是否开启模型训练日志可视化服务，默认为False|
|save_inference|Boolean|是否需要保存推理可以格式的模型文件，默认为True|
|need_convert|Boolean|是否需要将模型格式转化为Serving可用的格式，默认为True|

**ModelHub API 相关的字段：**

|字段 |类型 |说明 |
|---|---|---|
|name|String|模型名称，如: pp-ocr|
|namespace|String|命名空间，默认为kubeflow|
|action|String|对自定义资源的操作，apply/create/patch/delete之一|
|endpoint|String|模型中心的URI，默认为http://minio-service.kubeflow:9000|
|model_name|String|模型名称，与 Training API 的模型名称要保持一致|
|model_version|String|模型版本，默认为 latest|
|pvc_name|String|工作流共享盘的PVC名称|

**Serving API 相关的字段：**

|字段 |类型 |说明 |
|---|---|---|
|name|String|模型名称，如: pp-ocr|
|namespace|String|命名空间，默认为kubeflow|
|action|String|对自定义资源的操作，apply/create/patch/delete之一|
|image|String|包含 Paddle Serving 依赖的模型服务镜像|
|model_name|String|模型名称，与 Training API 的模型名称要保持一致|
|model_version|String|模型版本，与 ModelHub API 中的模型版本要保持一致|
|endpoint|String|模型中心的URI，默认为http://minio-service.kubeflow:9000|
|port|Integer|模型服务的端口号，默认为8040|