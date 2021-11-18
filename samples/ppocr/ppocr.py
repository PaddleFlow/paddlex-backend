import os

import kfp
import kfp.dsl as dsl
from kfp import components
from kfp.onprem import use_k8s_secret

NAMESPACE = "kubeflow"

# model name and file
MODEL_NAME = "ppocr-det"
MODEL_VERSION = "latest"
MODEL_FILE = "wide-deep.tar.gz"
TRAIN_EPOCH = 4

# data source and storage
DATA_SOURCE_SECRET_NAME = "data-source"
DATA_CENTER_SECRET_NAME = "data-center"
DATA_SOURCE_URI = "bos://paddleflow-public.hkg.bcebos.com/icdar2015/"

# SampleSet
SAMPLE_SET_NAME = "icdar2015"
SAMPLE_SET_PATH = "/mnt/icdar2015"

# config file and path
CONFIG_PATH = "/mnt/config/"
CONFIG_FILE = "det_mv3_db.yml"

# PaddleJob
TASK_MOUNT_PATH = "/mnt/task-center/"
MODEL_PATH = "/mnt/task-center/models/"
PADDLE_JOB_IMAGE = "registry.baidubce.com/paddleflow-public/paddleocr:2.1.0-gpu-cuda10.2-cudnn7"

# minio host
MINIO_ENDPOINT = "http://minio-service.kubeflow:9000"

# Serving Image
SERVING_IMAGE = "registry.baidubce.com/paddleflow-public/serving"


def create_volume_op():
    return dsl.VolumeOp(
        name="PPOCR Detection PVC",
        resource_name="ppocr-detection-pvc",
        storage_class="task-center",
        size="10Gi",
        modes=dsl.VOLUME_MODE_RWM
    ).set_display_name("create pvc and pv for PaddleJob"
    ).add_pod_annotation(name="pipelines.kubeflow.org/max_cache_staleness", value="P0D")


def create_resource_op():
    data_source_secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": DATA_SOURCE_SECRET_NAME,
            "namespace": NAMESPACE,
        },
        "type": "Opaque",
        "data": {
            "name": "ZW1wdHkgc2VjcmV0Cg=="
        }
    }

    return dsl.ResourceOp(
        name="Data Source Secret",
        action='apply',
        k8s_resource=data_source_secret,
    ).set_display_name("create data source secret for SampleJob"
    ).add_pod_annotation(name="pipelines.kubeflow.org/max_cache_staleness", value="P0D")


def create_sample_set():
    sampleset_launcher_op = components.load_component_from_file("../../components/sampleset.yaml")

    return sampleset_launcher_op(
        name=SAMPLE_SET_NAME,
        namespace=NAMESPACE,
        action="apply",
        partitions=1,
        secret_ref={"name": DATA_CENTER_SECRET_NAME}
    ).set_display_name(f"create sample set {SAMPLE_SET_NAME}")


def create_model_config(volume_op):
    det_mv3_db = f"""
Global:
  use_gpu: true
  epoch_num: 1200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {MODEL_PATH}
  save_epoch_step: 1200
  # evaluation is run every 2000 iterations after the 0th iteration
  eval_batch_step: [0, 2000]
  cal_metric_during_train: False
  pretrained_model: ./pretrain_models/MobileNetV3_large_x0_5_pretrained
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_en/img_10.jpg
  save_res_path: ./output/det_db/predicts_db.txt

Architecture:
  model_type: det
  algorithm: DB
  Transform:
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: large
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: 0.001
  regularizer:
    name: 'L2'
    factor: 0

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {SAMPLE_SET_PATH}/{SAMPLE_SET_NAME}/icdar_c4_train_imgs/
    label_file_list:
      - {SAMPLE_SET_PATH}/{SAMPLE_SET_NAME}/train_icdar2015_label.txt
    ratio_list: [1.0]
    transforms:
      - DecodeImage: # load image
          img_mode: BGR
          channel_first: False
      - DetLabelEncode: # Class handling label
      - IaaAugment:
          augmenter_args:
            - { 'type': Fliplr, 'args': { 'p': 0.5 } }
            - { 'type': Affine, 'args': { 'rotate': [-10, 10] } }
            - { 'type': Resize, 'args': { 'size': [0.5, 3] } }
      - EastRandomCropData:
          size: [640, 640]
          max_tries: 50
          keep_ratio: true
      - MakeBorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - MakeShrinkMap:
          shrink_ratio: 0.4
          min_text_size: 8
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: 'hwc'
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask'] # the order of the dataloader list
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 16
    num_workers: 8
    use_shared_memory: False

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {SAMPLE_SET_PATH}/{SAMPLE_SET_NAME}/ch4_test_images/
    label_file_list:
      - {SAMPLE_SET_PATH}/{SAMPLE_SET_NAME}/test_icdar2015_label.txt
    transforms:
      - DecodeImage: # load image
          img_mode: BGR
          channel_first: False
      - DetLabelEncode: # Class handling label
      - DetResizeForTest:
          image_shape: [736, 1280]
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: 'hwc'
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 1 # must be 1
    num_workers: 8
    use_shared_memory: False
"""
    write_config = components.load_component_from_file("../../components/configure.yaml")
    write_config_op = write_config(path=CONFIG_PATH, filename=CONFIG_FILE, content=det_mv3_db)

    write_config_op.add_pvolumes({CONFIG_PATH: volume_op.volume})
    write_config_op.set_display_name("create config file")
    return write_config_op


def create_sample_job():
    samplejob_launcher_op = components.load_component_from_file("../../components/samplejob.yaml")

    sync_options = {
        "syncOptions": {
            "source": DATA_SOURCE_URI,
            "destination": SAMPLE_SET_NAME,
        }
    }
    return samplejob_launcher_op(
        name="icdar2015-sync",
        namespace=NAMESPACE,
        type="sync",
        delete_after_done=True,
        job_options=sync_options,
        sampleset_ref={"name": SAMPLE_SET_NAME},
        secret_ref={"name": DATA_SOURCE_SECRET_NAME}
    ).set_display_name("sync remote data to local")


def create_paddle_job(volume_op):
    paddlejob_launcher_op = components.load_component_from_file("../../components/paddlejob.yaml")

    container = {
        "name": "paddleocr",
        "image": PADDLE_JOB_IMAGE,
        "workingDir": "/home/PaddleRec/models/rank/wide_deep",
        "command": ["python3"],
        "args": [
            "-m", "paddle.distributed.launch",
            "--gpus", "0,1",
            "tools/train.py",
            "-c", f"{CONFIG_PATH}/{CONFIG_FILE}",
        ],
        "volumeMounts": [
            {
                "name": "dshm",
                "mountPath": "/dev/shm"
            },
            {
                "name": "task-volume",
                "mountPath": TASK_MOUNT_PATH,
            }
        ],
        "resources": {
            "limits": {
                "nvidia.com/gpu": 2
            }
        }
    }

    worker = {
        "replicas": 1,
        "template": {"spec": {
            "containers": [container],
            "volumes": [
                {
                    "name": "dshm",
                    "emptyDir": {
                        "medium": "Memory"
                    },
                },
                {
                    "name": "task-volume",
                    "persistentVolumeClaim": {
                        "claimName": volume_op.volume.persistent_volume_claim.claim_name
                    }
                }
            ]},
        }
    }

    return paddlejob_launcher_op(
        name="ppocr-det",
        namespace=NAMESPACE,
        delete_after_done=True,
        worker_spec=worker,
        sampleset_ref={
            "name": SAMPLE_SET_NAME,
            "mountPath": SAMPLE_SET_PATH,
        }
    ).set_display_name("train ppocr detection model")


def create_convert_op(volume_op):
    model_converter = components.load_component_from_file("../../components/paddlejob.yaml")
    convert_op = model_converter(mount_path=TASK_MOUNT_PATH, model_name=MODEL_NAME, dirname=MODEL_PATH)
    convert_op.add_pvolumes({TASK_MOUNT_PATH: volume_op.volume})
    convert_op.set_display_name("Convert Model Format")
    return convert_op


def create_upload_op(volume_op):
    uploader = components.load_component_from_file("../../components/uploader.yaml")
    uploader_op = uploader(
        endpoint=MINIO_ENDPOINT,
        model_file=f"{TASK_MOUNT_PATH}{MODEL_NAME}.tar.gz",
        model_name=MODEL_NAME,
        version=MODEL_VERSION
    )

    uploader_op.add_pvolumes({TASK_MOUNT_PATH: volume_op.volume})
    uploader_op.apply(
        use_k8s_secret(
            secret_name=DATA_CENTER_SECRET_NAME,
            k8s_secret_key_to_env={
                "secret-key": "MINIO_SECRET_KEY",
                "access-key": "MINIO_ACCESS_KEY",
            },
        )
    )
    uploader_op.set_display_name("upload model to model-center")

    return uploader_op


def create_serving_op():
    create_serving = components.load_component_from_file("../../components/serving.yaml")

    args = f"wget {MINIO_ENDPOINT}/model-center/{MODEL_NAME}/{MODEL_VERSION}/{MODEL_NAME}.tar.gz && " \
           f"tar xzf {MODEL_NAME}.tar.gz && rm -rf {MODEL_NAME}.tar.gz && " \
           f"python3 -m paddle_serving_server.serve --model {MODEL_NAME}/server --port 9292"

    default = {
        "arg": args,
        "port": 9292,
        "tag": "v0.6.2",
        "containerImage": SERVING_IMAGE,
    }

    serving_op = create_serving(
        name="ppocr-det-serving",
        namespace="kubeflow",
        action="apply",
        default=default,
        runtime_version="paddleserving",
        service={"minScale": 1}
    )
    serving_op.set_display_name("deploy ppocr det serving")
    return serving_op


@dsl.pipeline(
    name="ppocr-detection-demo",
    description="An example for using ppocr train .",
)
def ppocr_detection_demo():
    # 1. create volume for config task and PaddleJob
    volume_op = create_volume_op()

    # 1. create secret for data source
    secret_op = create_resource_op()

    # 2. create or update SampleSet
    sample_set_task = create_sample_set()

    # 2. create configmap for model
    create_config_task = create_model_config(volume_op)
    create_config_task.after(volume_op)
    create_config_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # 3. create SampleJob and wait it finish data synchronization
    sample_job_task = create_sample_job()
    sample_job_task.after(secret_op, sample_set_task)
    sample_job_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # 4. create PaddleJob and wait it finish model training
    paddle_job_task = create_paddle_job(volume_op)
    paddle_job_task.after(create_config_task, sample_job_task)
    paddle_job_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # 5. convert model format and generate client/server proto file
    convert_op = create_convert_op(volume_op)
    convert_op.after(paddle_job_task)
    convert_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # 6. pack and compress model file then upload it to model-center
    upload_op = create_upload_op(volume_op)
    upload_op.after(convert_op)
    upload_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # volume_op.delete()

    # 7. download model file and deploy PaddleServing
    serving_op = create_serving_op()
    serving_op.after(upload_op)
    serving_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


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
    print(f"Created run {run}")