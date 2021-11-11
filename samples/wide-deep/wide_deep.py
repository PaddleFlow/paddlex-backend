import os

import kfp
import kfp.dsl as dsl
from kfp import components

from kubernetes.client.models import V1Volume, V1VolumeMount

NAMESPACE = "kubeflow"

# model name and file
MODEL_NAME = "wide-deep"
MODEL_VERSION = "latest"
MODEL_FILE = "wide-deep.tar.gz"

# SampleSet
SAMPLE_SET_NAME = "criteo"
SAMPLE_SET_PATH = "/mnt/criteo"

# data source and storage
DATA_SOURCE_SECRET_NAME = "data-source"
DATA_CENTER_SECRET_NAME = "data-center"
DATA_SOURCE_URI = "bos://paddleflow-public/criteo/demo/"
DATA_DESTINATION = "criteo/slot_train_data_full/"

# config file and path
CONFIG_PATH = "/mnt/config/"
CONFIG_FILE = "wide_deep_config.yaml"

# model file
# MODEL_CENTER = "/mnt/model-center/"
MODEL_PATH = "/mnt/model-center/PaddleRec/wide-deep/"

# PaddleJob
TASK_MOUNT_PATH = "/mnt/task-center/"
FLEET_LOG_PATH = "/mnt/task-center/logs/"

PADDLE_JOB_IMAGE = "registry.baidubce.com/paddleflow-public/paddlerec:2.1.0-gpu-cuda10.2-cudnn7"


def create_sample_set():
    sampleset_launcher_op = components.load_component_from_file("../../components/sampleset.yaml")

    return sampleset_launcher_op(
        name=SAMPLE_SET_NAME,
        namespace=NAMESPACE,
        action="apply",
        partitions=1,
        secret_ref={"name": DATA_CENTER_SECRET_NAME}
    ).set_display_name(f"create sample set {SAMPLE_SET_NAME}")


def create_sample_job():
    samplejob_launcher_op = components.load_component_from_file("../../components/samplejob.yaml")

    sync_options = {
        "syncOptions": {
            "source": DATA_SOURCE_URI,
            "destination": DATA_DESTINATION,
        }
    }
    return samplejob_launcher_op(
        name="criteo-sync",
        namespace=NAMESPACE,
        type="sync",
        delete_after_done=True,
        job_options=sync_options,
        sampleset_ref={"name": SAMPLE_SET_NAME},
        secret_ref={"name": DATA_SOURCE_SECRET_NAME}
    ).set_display_name("sync remote data to local")


def write_config_file(path: str, filename: str, content: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as f:
        f.write(content)


def create_model_config(volume_op):
    wide_deep_config = f"""
# wide_deep_config.yaml
# global settings
runner:
  train_data_dir: "{SAMPLE_SET_PATH}/{DATA_DESTINATION}"
  train_reader_path: "criteo_reader" # importlib format
  use_gpu: True
  use_auc: True
  train_batch_size: 4096
  epochs: 4
  print_interval: 10
  model_save_path: {MODEL_PATH}
  test_data_dir: "/mnt/{SAMPLE_SET_NAME}/slot_test_data_full"
  infer_reader_path: "criteo_reader" # importlib format
  infer_batch_size: 4096
  infer_load_path: {MODEL_PATH}
  infer_start_epoch: 0
  infer_end_epoch: 4
  use_inference: True
  save_inference_feed_varnames: ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","dense_input"]
  save_inference_fetch_varnames: ["sigmoid_0.tmp_0"]
  #use fleet
  use_fleet: True

# hyper parameters of user-defined network
hyper_parameters:
  # optimizer config
  optimizer:
    class: Adam
    learning_rate: 0.001
    strategy: async
  # user-defined <key, value> pairs
  sparse_inputs_slots: 27
  sparse_feature_number: 1000001
  sparse_feature_dim: 9
  dense_input_dim: 13
  fc_sizes: [512, 256, 128, 32]
  distributed_embedding: 0
"""

    create_config = components.create_component_from_func(
        func=write_config_file,
        base_image="python:3.7",
    )
    task_op = create_config(CONFIG_PATH, CONFIG_FILE, wide_deep_config)
    task_op.add_pvolumes({CONFIG_PATH: volume_op.volume})
    task_op.set_display_name("train wide and deep")

    return task_op


def create_paddle_job(volume_op):
    paddlejob_launcher_op = components.load_component_from_file("../../components/paddlejob.yaml")
    args = f"cp {os.path.join(TASK_MOUNT_PATH, CONFIG_FILE)} . &&  " \
           f"python -m paddle.distributed.launch --log_dir {FLEET_LOG_PATH} ../../../tools/trainer.py -m {CONFIG_FILE}"

    container = {
        "name": "paddlerec",
        "image": PADDLE_JOB_IMAGE,
        "workingDir": "/home/PaddleRec/models/rank/wide_deep",
        "command": ["/bin/bash"],
        "args": [
            "-c", args
        ],
        "volumeMounts": [
            {
                "name": "dshm",
                "mountPath": "/dev/shm"
            },
            {
                "name": "task-volume",
                "mountPath": volume_op.volume.name,
            }
        ],
        "resources": {
            "limits": {
                "nvidia.com/gpu": 1
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
                        "claimName": "model-center"
                    }
                }
            ]},
        }
    }

    return paddlejob_launcher_op(
        name="wide-and-deep",
        namespace=NAMESPACE,
        delete_after_done=True,
        worker_spec=worker,
        sampleset_ref={
            "name": SAMPLE_SET_NAME,
            "mountPath": SAMPLE_SET_PATH,
        }
    ).set_display_name("train wide and deep")


def create_uploader_op():
    pass


def create_volume_op():
    return dsl.VolumeOp(
        name="Wide Deep PVC",
        resource_name="wide-deep-pvc",
        storage_class="task-center",
        size="10Gi",
        modes=dsl.VOLUME_MODE_RWM
    ).set_display_name("create pvc and pv for PaddleJob")


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
    ).set_display_name("create data source secret for SampleJob")


def create_upload_op(volume_op):
    uploader = components.load_component_from_file("../../components/uploader.yaml")
    uploader_op = uploader(model_name=MODEL_NAME, version=MODEL_VERSION, filename=MODEL_FILE)

    uploader_op.add_volume(V1Volume(
        name="config-center",
        persistent_volume_claim={
            "claimName": "config-center"
        }
    ))

    uploader_op.add_volume_mount(

    )
    uploader_op.set_display_name("train wide and deep")
    return uploader_op


def create_serving_op():
    create_serving = components.load_component_from_file("../../components/serving.yaml")


@dsl.pipeline(
    name="paddle-wide-deep",
    description="An example to train wide-deep-deep using paddle.",
)
def wide_deep_demo():
    # 1. create volume for config task and PaddleJob
    volume_op = create_volume_op()

    # 1. create secret for data source
    secret_op = create_resource_op()

    # 2. create or update SampleSet for Criteo which is stored in remote storage
    sample_set_task = create_sample_set()

    # 2. create configmap for wide-and-deep model
    create_config_task = create_model_config(volume_op)
    create_config_task.after(volume_op)

    # 3. create SampleJob and wait it finish data synchronization
    sample_job_task = create_sample_job()
    sample_job_task.after(secret_op, sample_set_task)

    # 4. create PaddleJob and wait it finish model training
    paddle_job_task = create_paddle_job(volume_op)
    paddle_job_task.after(create_config_task, sample_job_task)

    # # 5. pack and compress model file then upload it to model-center
    # upload_op = create_upload_op(volume_op)
    #
    # # 6. download model file and deploy PaddleServing


if __name__ == "__main__":
    import kfp.compiler as compiler

    pipeline_file = "wide_deep_demo.yaml"

    compiler.Compiler().compile(wide_deep_demo, pipeline_file)

    client = kfp.Client(host="http://www.my-pipeline-ui.com:80")
    run = client.create_run_from_pipeline_package(
        pipeline_file,
        arguments={},
        run_name="paddle wide-and-deep demo",
        service_account="paddle-operator"
    )
    print(f"Created run {run}")
