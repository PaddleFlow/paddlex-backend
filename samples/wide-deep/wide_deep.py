import os

import kfp
import kfp.dsl as dsl
from kfp import components
from kfp.onprem import use_k8s_secret

NAMESPACE = "kubeflow"

# model name and file
MODEL_NAME = "wide-deep"
MODEL_VERSION = "latest"
MODEL_FILE = "wide-deep.tar.gz"
TRAIN_EPOCH = 4

# data source and storage
DATA_CENTER_NAME = "data-center"
DATA_SOURCE_SECRET_NAME = "data-source"
DATA_CENTER_SECRET_NAME = "data-center"
DATA_SOURCE_URI = "bos://paddleflow-public.hkg.bcebos.com/criteo/demo/"

# SampleSet
DATA_SET_NAME = "criteo"
DATA_MOUNT_PATH = f"/mnt/{DATA_CENTER_NAME}"

# config file
CONFIG_FILE = "wide_deep_config.yaml"

# PaddleJob
TASK_MOUNT_PATH = "/mnt/task-center/"
MODEL_CHECKPOINT_PATH = "/mnt/task-center/models/"

# minio host
MINIO_ENDPOINT = "http://minio-service.kubeflow:9000"

# images
PADDLE_JOB_IMAGE = "registry.baidubce.com/paddleflow-public/paddlerec:2.1.0-gpu-cuda10.2-cudnn7"
SERVING_IMAGE = "registry.baidubce.com/paddleflow-public/serving"


def create_sample_set():
    sampleset_launcher_op = components.load_component_from_file("../../components/sampleset.yaml")

    return sampleset_launcher_op(
        name=DATA_CENTER_NAME,
        namespace=NAMESPACE,
        action="apply",
        partitions=1,
        secret_ref={"name": DATA_CENTER_SECRET_NAME}
    ).set_display_name(f"create sample set")


def create_sample_job():
    samplejob_launcher_op = components.load_component_from_file("../../components/samplejob.yaml")

    sync_options = {
        "syncOptions": {
            "source": DATA_SOURCE_URI,
            "destination": DATA_SET_NAME,
        }
    }
    return samplejob_launcher_op(
        name="criteo-sync",
        namespace=NAMESPACE,
        type="sync",
        delete_after_done=True,
        job_options=sync_options,
        sampleset_ref={"name": DATA_CENTER_NAME},
        secret_ref={"name": DATA_SOURCE_SECRET_NAME}
    ).set_display_name("sync remote data to local")


def write_config_file(path: str, filename: str, content: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as f:
        f.write(content)

    print("write config success: \n\n {}".format(content))


def create_model_config(volume_op):
    wide_deep_config = f"""
# wide_deep_config.yaml
# global settings
runner:
  train_data_dir: "{DATA_MOUNT_PATH}/{DATA_SET_NAME}/slot_train_data_full/"
  train_reader_path: "criteo_reader" # importlib format
  use_gpu: True
  use_auc: True
  train_batch_size: 4096
  epochs: {TRAIN_EPOCH}
  print_interval: 10
  model_save_path: {MODEL_CHECKPOINT_PATH}
  test_data_dir: "{DATA_MOUNT_PATH}/{DATA_SET_NAME}/slot_test_data_full"
  infer_reader_path: "criteo_reader" # importlib format
  infer_batch_size: 4096
  infer_load_path: {MODEL_CHECKPOINT_PATH}
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
    task_op = create_config(TASK_MOUNT_PATH, CONFIG_FILE, wide_deep_config)
    task_op.add_pvolumes({TASK_MOUNT_PATH: volume_op.volume})
    task_op.set_display_name("create config file")

    return task_op


def create_paddle_job(volume_op):
    paddlejob_launcher_op = components.load_component_from_file("../../components/paddlejob.yaml")
    args = f"cp {os.path.join(TASK_MOUNT_PATH, CONFIG_FILE)} . &&  " \
           f"python -m paddle.distributed.launch --log_dir {TASK_MOUNT_PATH} ../../../tools/static_trainer.py -m {CONFIG_FILE}"

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
                "mountPath": TASK_MOUNT_PATH,
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
                        "claimName": volume_op.volume.persistent_volume_claim.claim_name
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
            "name": DATA_CENTER_NAME,
            "mountPath": DATA_MOUNT_PATH,
        }
    ).set_display_name("train wide and deep")


def create_volume_op():
    return dsl.VolumeOp(
        name="Wide Deep PVC",
        resource_name="wide-deep-pvc",
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


def create_convert_op(volume_op):
    args = f"cd {TASK_MOUNT_PATH} && mkdir -p {MODEL_NAME} && " \
           f"python3 -m paddle_serving_client.convert --dirname {MODEL_CHECKPOINT_PATH}{TRAIN_EPOCH-1}/ " \
           f"--model_filename rec_inference.pdmodel --params_filename rec_inference.pdiparams " \
           f"--serving_server ./{MODEL_NAME}/server/ --serving_client ./{MODEL_NAME}/client/ &&" \
           f"tar czf {MODEL_NAME}.tar.gz {MODEL_NAME}/"

    convert_op = dsl.ContainerOp(
        name="Convert wide&Deep Model",
        image=f"{SERVING_IMAGE}:v0.6.2",
        command="/bin/bash",
        arguments=["-c", args],
    )
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
        name="wide-deep-serving",
        namespace="kubeflow",
        action="apply",
        default=default,
        runtime_version="paddleserving",
        service={"minScale": 1}
    )
    serving_op.set_display_name("deploy wide-deep serving")
    return serving_op


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
    # sample_set_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # 2. create configmap for wide-and-deep model
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

    # 8. get predict ctr from wide & deep model server


if __name__ == "__main__":
    import kfp.compiler as compiler

    pipeline_file = "wide_deep_demo.yaml"

    compiler.Compiler().compile(wide_deep_demo, pipeline_file)

    client = kfp.Client(host="http://www.my-pipeline-ui.com:80")
    run = client.create_run_from_pipeline_package(
        pipeline_file,
        arguments={},
        run_name="paddle wide-and-deep demo",
        service_account="pipeline-runner"
    )
    print(f"Created run {run}")
