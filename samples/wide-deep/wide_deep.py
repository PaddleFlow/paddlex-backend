import os

import kfp
import kfp.dsl as dsl
from kfp import components

from kubernetes.client.models import V1Volume, V1VolumeMount


NAMESPACE = "kubeflow"

# SampleSet
SAMPLE_SET_NAME = "criteo"
SAMPLE_SET_PATH = "/mnt/criteo"

DATA_SOURCE_SECRET = "data-source-secret"
DATA_CENTER_SECRET = "data-center-secret"

DATA_SOURCE_URI = "bos://paddleflow-public/criteo/slot_train_data_full/"
DATA_DESTINATION = "slot_train_data_full/"


# config file
CONFIG_CENTER = "/mnt/config-center/"
CONFIG_PATH = "/mnt/config-center/PaddleRec/"
CONFIG_FILE = "wide_deep_config.yaml"

# model file
MODEL_CENTER = "/mnt/model-center/"
MODEL_PATH = "/mnt/model-center/PaddleRec/wide-deep/"

# log file
LOG_PATH = "/mnt/model-center/PaddleRec/wide-deep/log"

PADDLE_JOB_IMAGE = "registry.baidubce.com/paddleflow-public/paddlerec:2.1.0-gpu-cuda10.2-cudnn7"


WIDE_DEEP_CONFIG = f"""
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

CMD = f"cp {os.path.join(CONFIG_PATH, CONFIG_FILE)} . &&  python -m paddle.distributed.launch --log_dir {LOG_PATH} ../../../tools/trainer.py -m {CONFIG_FILE}"


def create_sample_set():
    sampleset_launcher_op = components.load_component_from_file("./components/sampleset-component.yaml")

    return sampleset_launcher_op(
        name=SAMPLE_SET_NAME,
        namespace=NAMESPACE,
        partitions=1,
        secret_ref={"name": DATA_CENTER_SECRET}
    ).set_display_name(f"create sample set {SAMPLE_SET_NAME}")


def create_sample_job(sample_set_task):
    samplejob_launcher_op = components.load_component_from_file("./components/samplejob-component.yaml")

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
        secret_ref={"name": DATA_SOURCE_SECRET}
    ).after(sample_set_task).set_display_name("sync remote data to local")


def write_config_file(path: str, filename: str, content: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as f:
        f.write(content)


def create_model_config():
    create_config = components.create_component_from_func(
        func=write_config_file,
        base_image="python:3.7",
    )
    task_op = create_config(CONFIG_PATH, CONFIG_FILE, WIDE_DEEP_CONFIG)

    task_op.add_volume(V1Volume(
        name="config-center",
        persistent_volume_claim={
            "claimName": "config-center"
        }
    ))
    task_op.container.add_volume_mount(V1VolumeMount(
        name="config-center",
        mount_path=CONFIG_CENTER
    ))
    return task_op


def create_paddle_job(sample_job_task, model_config_task):
    paddlejob_launcher_op = components.load_component_from_file("./components/paddlejob-component.yaml")

    container = {
        "name": "paddlerec",
        "image": PADDLE_JOB_IMAGE,
        "workingDir": "/home/PaddleRec/models/rank/wide_deep",
        "command": ["/bin/bash"],
        "args": [
            "-c", CMD
        ],
        "volumeMounts": [
            {
                "name": "dshm",
                "mountPath": "/dev/shm"
            },
            {
                "name": "model-center",
                "mountPath": MODEL_CENTER,
            },
            {
                "name": "config-center",
                "mountPath": CONFIG_CENTER,
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
                    "name": "model-center",
                    "persistentVolumeClaim": {
                        "claimName": "model-center"
                    }
                },
                {
                    "name": "config-center",
                    "persistentVolumeClaim": {
                        "claimName": "config-center"
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
    ).after(sample_job_task, model_config_task
            ).set_display_name("train wide and deep")


@dsl.pipeline(
    name="paddle-wide-deep",
    description="An example to train wide-deep-deep using paddle.",
)
def wide_deep_demo():

    dsl.VolumeOp()

    sample_set_task = create_sample_set()
    create_config_task = create_model_config()
    sample_job_task = create_sample_job(sample_set_task)
    paddle_job_task = create_paddle_job(sample_job_task, create_config_task)


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
