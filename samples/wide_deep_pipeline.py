import kfp
import json
import os
from kfp.onprem import use_k8s_secret
from kfp import components
from kfp.components import load_component_from_file
from kfp import dsl
from kfp import compiler

# 1. 准备数据 - paddle-launcher 提交 SampleJob 同步数据
# 2. 模型训练 - paddle-launcher 提交 PaddleJob 训练模型
# 3. 模型转换 -
# 4. 模型服务
# 5. 测试模型服务

# image: library/bash:4.4.23
# image: bitnami/kubectl:1.17.17

# 定义全局常量

NAMESPACE = "kubeflow"

# COOKIE="authservice_session="+AUTH

EXPERIMENT = "Default"

dist_volume = 'dist-vol'

volume_mount_path = "/model"

dataset_path = volume_mount_path+"/dataset"

checkpoint_dir = volume_mount_path+"/checkpoint"

tensorboard_root = volume_mount_path+"/tensorboard"

# Set Log bucket and Tensorboard Image

MINIO_ENDPOINT = "http://localhost:9000"

LOG_BUCKET = "mlpipeline"

TENSORBOARD_IMAGE = "public.ecr.aws/pytorch-samples/tboard:latest"

# client = kfp.Client(host=INGRESS_GATEWAY+"/pipeline", cookies=COOKIE)
client = kfp.Client(host="http://www.my-pipeline-ui.com:80")

client.create_experiment(EXPERIMENT)
experiments = client.list_experiments()
my_experiment = experiments.experiments[0]

print(my_experiment)

DEPLOY_NAME = "bert-dist"
MODEL_NAME = "bert"


prep_op = components.load_component_from_file(
    "yaml/preprocess_component.yaml"
)

# Use GPU image in train component
# train_op = components.load_component_from_file(
#     "yaml/train_component.yaml"
# )

deploy_op = load_component_from_file(
    "yaml/deploy_component.yaml"
)


minio_op = components.load_component_from_file(
    "yaml/minio_component.yaml"
)


pytorch_job_op = load_component_from_file("../../../components/kubeflow/pytorch-launcher/component.yaml")

kubernetes_create_pvc_op = load_component_from_file(
    "../../../components/kubernetes/Create_PersistentVolumeClaim/component.yaml"
)

cp_op = load_component_from_file(
    "yaml/copy_component.yaml"
)


from kubernetes.client.models import V1Volume, V1PersistentVolumeClaimVolumeSource

def create_dist_pipeline():
    kubernetes_create_pvc_op(name=dist_volume, storage_size="20Gi")


print("===== create pvc ======")
create_volume_run = client.create_run_from_pipeline_func(create_dist_pipeline, arguments={})

create_volume_run.wait_for_run_completion()
print("===== create pvc finish======")
# Define pipeline


@dsl.pipeline(name="Training pipeline", description="Sample training job test")
def pytorch_bert(
        minio_endpoint=MINIO_ENDPOINT,
        log_bucket=LOG_BUCKET,
        log_dir=f"tensorboard/logs/{dsl.RUN_ID_PLACEHOLDER}",
        confusion_matrix_log_dir=f"confusion_matrix/{dsl.RUN_ID_PLACEHOLDER}/",
        mar_path=f"mar/{dsl.RUN_ID_PLACEHOLDER}/model-store/",
        config_prop_path=f"mar/{dsl.RUN_ID_PLACEHOLDER}/config/",
        model_uri=f"pvc://{dist_volume}/mar/{dsl.RUN_ID_PLACEHOLDER}",
        tf_image=TENSORBOARD_IMAGE,
        deploy=DEPLOY_NAME,
        namespace=NAMESPACE,
        num_samples=1000,
        max_epochs=1,
        gpus=2,
        num_nodes=2
):

    prepare_tb_task = prepare_tensorboard_op(
        log_dir_uri=f"s3://{log_bucket}/{log_dir}",
        image=tf_image,
        pod_template_spec=json.dumps({
            "spec": {
                "containers": [{
                    "env": [
                        {
                            "name": "AWS_ACCESS_KEY_ID",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "mlpipeline-minio-artifact",
                                    "key": "accesskey",
                                }
                            },
                        },
                        {
                            "name": "AWS_SECRET_ACCESS_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "mlpipeline-minio-artifact",
                                    "key": "secretkey",
                                }
                            },
                        },
                        {
                            "name": "AWS_REGION",
                            "value": "minio"
                        },
                        {
                            "name": "S3_ENDPOINT",
                            "value": f"{minio_endpoint}",
                        },
                        {
                            "name": "S3_USE_HTTPS",
                            "value": "0"
                        },
                        {
                            "name": "S3_VERIFY_SSL",
                            "value": "0"
                        },
                    ]
                }]
            }
        }),
    ).set_display_name("Visualization")

    prep_task = prep_op().after(prepare_tb_task).set_display_name("Preprocess & Transform")

    # prep_task = prep_op().set_display_name("Preprocess & Transform")

    copy_task = cp_op(
        "true",
        prep_task.outputs['output_data'],
        dataset_path,
        ""
    ).add_pvolumes(
        {
            volume_mount_path: dsl.PipelineVolume(pvc=dist_volume)
        }
    ).after(prep_task).set_display_name("Copy Dataset")

    confusion_matrix_url = f"minio://{log_bucket}/{confusion_matrix_log_dir}"

    train_task = pytorch_job_op(
        name="pytorch-bert",
        namespace=namespace,
        master_spec=
        {
            "replicas": 1,
            "imagePullPolicy": "Always",
            "restartPolicy": "OnFailure",
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "pytorch",
                            "image": "public.ecr.aws/pytorch-samples/kfp_samples:latest-gpu",
                            "command": ["python3", "bert/agnews_classification_pytorch.py"],
                            "args": [
                                "--dataset_path", dataset_path,
                                "--checkpoint_dir", checkpoint_dir,
                                "--script_args", f"model_name=bert.pth,num_samples={num_samples}",
                                "--tensorboard_root", tensorboard_root,
                                "--ptl_args", f"max_epochs={max_epochs},profiler=pytorch,gpus={gpus},accelerator=ddp,num_nodes={num_nodes},confusion_matrix_url={confusion_matrix_url}"
                            ],
                            "ports": [
                                {
                                    "containerPort": 24456,
                                    "name": "pytorchjob-port"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": 2
                                }
                            },
                            "volumeMounts": [
                                {
                                    "mountPath": volume_mount_path,
                                    "name": "model-volume"
                                }
                            ]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "model-volume",
                            "persistentVolumeClaim": {
                                "claimName": dist_volume
                            }
                        }
                    ]
                }
            }
        },
        worker_spec=
        {
            "replicas": 1,
            "imagePullPolicy": "Always",
            "restartPolicy": "OnFailure",
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "pytorch",
                            "image": "public.ecr.aws/pytorch-samples/kfp_samples:latest-gpu",
                            "command": ["python3", "bert/agnews_classification_pytorch.py"],
                            "args": [
                                "--dataset_path", dataset_path,
                                "--checkpoint_dir", checkpoint_dir,
                                "--script_args", f"model_name=bert.pth,num_samples={num_samples}",
                                "--tensorboard_root", tensorboard_root,
                                "--ptl_args", f"max_epochs={max_epochs},profiler=pytorch,gpus={gpus},accelerator=ddp,num_nodes={num_nodes},confusion_matrix_url={confusion_matrix_url}"
                            ],
                            "ports": [
                                {
                                    "containerPort": 24456,
                                    "name": "pytorchjob-port"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": 2
                                }
                            },
                            "volumeMounts": [
                                {
                                    "mountPath": volume_mount_path,
                                    "name": "model-volume"
                                }
                            ]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "model-volume",
                            "persistentVolumeClaim": {
                                "claimName": dist_volume
                            }
                        }
                    ]
                }
            }
        },
        delete_after_done=False
    ).after(copy_task)

    mar_folder_restructure_task = dsl.ContainerOp(
        name='mar restructure',
        image='library/bash:4.4.23',
        command=['sh', '-c'],
        arguments=[f'mkdir -p {volume_mount_path}/{mar_path}; mkdir -p {volume_mount_path}/{config_prop_path}; cp {checkpoint_dir}/*.mar {volume_mount_path}/{mar_path}; cp {checkpoint_dir}/config.properties {volume_mount_path}/{config_prop_path}']
    ).add_pvolumes(
        {volume_mount_path: dsl.PipelineVolume(pvc=dist_volume)}
    ).after(train_task).set_display_name("Restructure MAR and config.properties path")

    mar_folder_restructure_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    copy_tensorboard = cp_op("false", "", "", tensorboard_root).add_pvolumes({volume_mount_path: dsl.PipelineVolume(pvc=dist_volume)}).after(mar_folder_restructure_task).set_display_name("Copy Tensorboard Logs")

    copy_tensorboard.execution_options.caching_strategy.max_cache_staleness = "P0D"

    minio_tb_upload = (
        minio_op(
            bucket_name=log_bucket,
            folder_name=log_dir,
            input_path=copy_tensorboard.outputs["destination_path"],
            filename="",
        ).after(copy_tensorboard)
            .set_display_name("Tensorboard Events Pusher")
    )

    # Deploy inferenceservice in gpu
    gpu_count = "1"

    isvc_gpu_yaml = """
    apiVersion: "serving.kubeflow.org/v1beta1"
    kind: "InferenceService"
    metadata:
      name: {}
      namespace: {}
    spec:
      predictor:
        serviceAccountName: sa
        pytorch:
          storageUri: {}
          resources:
            requests: 
              cpu: 4
              memory: 8Gi
            limits:
              cpu: 4
              memory: 8Gi
              nvidia.com/gpu: {}
    """.format(
        deploy, namespace, model_uri, gpu_count
    )

    deploy_task = (
        deploy_op(action="apply", inferenceservice_yaml=isvc_gpu_yaml)
            .after(minio_tb_upload)
            .set_display_name("Deployer")
    )

    deploy_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    dsl.get_pipeline_conf().add_op_transformer(
        use_k8s_secret(
            secret_name="mlpipeline-minio-artifact",
            k8s_secret_key_to_env={
                "secretkey": "MINIO_SECRET_KEY",
                "accesskey": "MINIO_ACCESS_KEY",
            },
        )
    )


# Compile pipeline
print("===== Compile pipeline ======")
compiler.Compiler().compile(pytorch_bert, 'pytorch.yaml', type_check=True)

# Execute pipeline
print("===== Execute pipeline ======")
# run = client.run_pipeline(my_experiment.id, 'pytorch-bert', 'pytorch.tar.gz')
