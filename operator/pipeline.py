import kfp
import kfp.dsl as dsl
from kfp import components


def create_volume_op():
    return dsl.VolumeOp(
        name="PPOCR Detection PVC",
        resource_name="ppocr-detection-pvc",
        storage_class="task-center",
        size="10Gi",
        modes=dsl.VOLUME_MODE_RWM
    )


def create_dataset_op(volume_op):
    """

    :param volume_op:
    :return:
    """
    sampleset_op = components.load_component_from_file("dataset.yaml")
    return sampleset_op(
        name="icdar2015",
        partitions=1,
        secret="icdar2015",
        pvc_name=volume_op.volume.persistent_volume_claim.claim_name,
        source_uri="bos://paddleflow-public.hkg.bcebos.com/icdar2015/"
    )


def create_training_op(volume_op):
    training_op = components.load_component_from_file("traning.yaml")
    return training_op(
        name="ppocr",
        sampleSetRef="icdar2015",
        mode="collective",
        replicas=1,
        visualdl=True,
        pvc_name=volume_op.volume.persistent_volume_claim.claim_name,
        image="registry.baidubce.com/paddleflow-public/paddleocr:2.1.3-gpu-cuda10.2-cudnn7",
        command="python -m paddle.distributed.launch --gpus '0' tools/train.py -c det_mv3_db.yml"
    )


def create_modelhub_op(volume_op):
    modelhub_op = components.load_component_from_file("modelhub.yaml")
    return modelhub_op(
        name="ppocr",
        target="minio://modelhub/ppocr/",
        version="latest",
        convert="inference",
        pvc_name=volume_op.volume.persistent_volume_claim.claim_name,
    )


def create_serving_op():
    serving_op = components.load_component_from_file("serving.yaml")
    return serving_op(
        name="ppocr",
        port="9292",
        model_uri="minio://modelhub/ppocr/latest/",
        image="registry.baidubce.com/paddleflow-public/serving:v0.6.2",
        command="python3 -m paddle_serving_server.serve --model ppocr/server --port 9292"
    )


@dsl.pipeline(
    name="ppocr-detection-demo",
    description="An example for using ppocr train.",
)
def ppocr_detection_demo():
    # 创建 ppocr pipeline 各步骤所需的存储盘
    volume_op = create_volume_op()

    # 拉取远程数据（BOS/HDFS）到训练集群本地，并缓存
    sampleset_op = create_dataset_op(volume_op)
    sampleset_op.after(volume_op)

    # 采用Collective模型分布式训练ppocr模型，并提供模型训练可视化服务
    training_op = create_training_op(volume_op)
    training_op.after(sampleset_op)

    # 将模型转换为 PaddleServing 可用的模型格式，并上传到模型中心
    modelhub_op = create_modelhub_op(volume_op)
    modelhub_op.after(training_op)

    # 从模型中心下载模型，并启动 PaddleServing 服务
    serving_op = create_serving_op()
    serving_op.after(modelhub_op)


if __name__ == "__main__":
    import kfp.compiler as compiler

    pipeline_file = "ppocr_detection_demo.yaml"
    compiler.Compiler().compile(ppocr_detection_demo, pipeline_file)
    client = kfp.Client(host="http://www.my-pipeline-ui.com:80")
    run = client.create_run_from_pipeline_package(
        pipeline_file,
        arguments={},
        run_name="paddle ocr detection demo",
        service_account="pipeline-runner"
    )
