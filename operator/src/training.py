import argparse
import logging
import launch_crd
from launch_crd import logger
from distutils.util import strtobool

from kubernetes import client as k8s_client
from kubernetes.client import rest
from kubernetes import config


class PaddleJob(launch_crd.K8sCR):

    def __init__(self, client=None):
        super(PaddleJob, self).__init__("batch.paddlepaddle.org", "paddlejobs", "v1", client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("phase")
        if not conditions:
            return False, ""
        if conditions in expected_conditions:
            return True, conditions
        else:
            return False, conditions

    def get_command(self, spec):
        project = spec.get("project")
        if project == "paddleocr":
            return self.get_ocr_command(spec)
        else:
            raise Exception(f"{project} is not supported now")

    def get_ocr_command(self, spec):
        command = ""

        pretrain_model = spec.get("pretrain_model", None)
        if pretrain_model is not None and len(pretrain_model) != 0:
            command += f"wget -P /mnt/task-center/pretrain_model/ {pretrain_model} && "

        command += "python "
        if spec.get("gpu_per_node") > 0:
            gpus = ",".join(str(i) for i in range(spec.get("gpu_per_node")))
            command += f"-m paddle.distributed.launch --gpus '{gpus}' --log_dir /mnt/task-center "
        command += f"tools/train.py -c {spec.get('config_path')} -o "

        config_changes = spec.get("config_changes")
        if config_changes:
            config_changes = config_changes.split(",")
            for change in config_changes:
                command += f"{change.strip(' ')} "

        command += "Global.save_model_dir=/mnt/task-center/models/ "
        command += "Global.save_inference_dir=/mnt/task-center/inference/ "
        if pretrain_model is not None and len(pretrain_model) != 0:
            command += f"Global.pretrained_model=/mnt/task-center/pretrain_model/{pretrain_model} "

        command += f"Train.dataset.data_dir=/mnt/data-center/{spec.get('dataset')}/ "
        if spec.get("train_label") is not None:
            command += f"""Train.dataset.label_file_list=["/mnt/data-center/{spec.get('dataset')}/{spec.get('train_label')}"] """
        command += f"Eval.dataset.data_dir=/mnt/data-center/{spec.get('dataset')}/ "
        if spec.get("test_label") is not None:
            command += f"""Eval.dataset.label_file_list=["/mnt/data-center/{spec.get('dataset')}/{spec.get('test_label')}"] """

        return command

    def get_spec(self, spec):

        volume_mounts = [
            {
                "name": "dshm",
                "mountPath": "/dev/shm"
            },
            {
                "name": "task-volume",
                "mountPath": "/mnt/task-center/",
            }
        ]

        container = {
            "name": spec.get("name"),
            "image": spec.get("image"),
            "command": ["/bin/bash"],
            "args": [
                "-c", spec.get("command")
            ],
        }

        paddlejob = {
            "apiVersion": "%s/%s" % (self.group, self.version),
            "kind": "PaddleJob",
            "metadata": {
                "name": spec.get("name"),
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "cleanPodPolicy": "OnCompletion",
                "withGloo": 1,
                "intranet": "PodIP",
                "sampleSetRef": {
                    "name": "data-center",
                    "mountPath": "/mnt/data-center"
                },
            },
        }

        if spec.get("worker_replicas", 0) > 0:
            worker_container = container.copy()
            worker_container["volumeMounts"] = volume_mounts
            if spec.get("gpu_per_node", None):
                worker_container["resources"] = {
                    "limits": {
                        "nvidia.com/gpu": spec.get("gpu_per_node")
                    }
                }
            paddlejob["worker"] = {
                "replicas": spec.get("worker_replicas"),
                "template": {"spec": {
                    "containers": [worker_container],
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
                                "claimName": spec.get("pvc_name"),
                            }
                        }
                    ]},
                }
            }

        if spec.get("ps_replicas", 0) > 0:
            paddlejob["ps"] = {
                "replicas": spec.get("ps_replicas"),
                "template": {
                    "spec": {
                        "containers": [container],
                    }
                }
            }

        return paddlejob

    def get_action_status(self, action=None):
        return ["Succeed", "Completed"], ["Failed", "Terminated"]


class VisualDL(launch_crd.K8sCR):
    def __init__(self, client=None):
        super(VisualDL, self).__init__("apps", "deployments", "v1", client)

    def get_action_status(self, action=None):
        return ["Available", "Progressing"], []

    def get_spec(self, spec):
        logdir = "/mnt/task-center/models/vdl/"
        model = f"/mnt/task-center/{spec.get('name')}/server/__model__"

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{spec.get('name')}-visualdl",
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "visualdl": f"{spec.get('name')}-visualdl",
                    }
                },
                "replicas": 1,
                "template": {
                    "metadata": {
                        "labels": {
                            "visualdl": f"{spec.get('name')}-visualdl",
                        },
                    },
                    "spec": {
                        "containers": [{
                            "name": "visualdl",
                            "image": "python:3.7",
                            "imagePullPolicy": "IfNotPresent",
                            "workingDir": "/mnt/task-center/",
                            "volumeMounts": [{
                                "name": "task-center",
                                "mountPath": "/mnt/task-center/",
                            }],
                            "command": ["/bin/bash"],
                            "args": ["-c", f"visualdl --logdir {logdir} --model {model}"],
                            "ports": [{
                                "name": "http",
                                "containerPort": 8040
                            }]
                        }],
                        "volumes": [{
                            "name": "task-center",
                            "persistentVolumeClaim": {
                                "claimName": spec.get("pvc_name"),
                            }
                        }]
                    }
                },
            },
        }


class JobOp(launch_crd.K8sCR):
    
    def __init__(self, client=None):
        super(JobOp, self).__init__("batch", "jobs", "v1", client)

    def get_action_status(self, action=None):
        return ["Complete"], ["Failed"]


class InferenceOp(JobOp):

    def get_spec(self, spec):
        container = {
            "image": spec.get("image"),
            "command": ["/bin/bash"],
            "volumeMounts": [{
                "name": "task-center",
                "mountPath": "/mnt/task-center/",
            }],
        }

        if spec.get("gpu_per_node", 0) > 0:
            container["resources"] = {
                "limits": {
                    "nvidia.com/gpu": spec.get("gpu_per_node")
                }
            }

        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"{spec.get('name')}-inference-convert",
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [container],
                        "volumes": [{
                            "name": "task-center",
                            "persistentVolumeClaim": {
                                "claimName": spec.get("pvc_name"),
                            }
                        }]
                    }
                }
            }
        }
        return job


class ConvertOp(JobOp):

    def get_spec(self, spec):
        mount_path = ""
        model_name = ""
        dirname = ""
        pdmodel = ""
        pdiparams = ""

        convert_shell = """
        mount_path=$0
        model_name=$1
        dirname=$2
        pdmodel=$3
        pdiparams=$4

        cd $mount_path && mkdir -p $model_name
        echo "mkdir dir ${model_name} successfully"

        python3 -m paddle_serving_client.convert --dirname $dirname --model_filename $pdmodel --params_filename $pdiparams --serving_server ./${model_name}/server/ --serving_client ./${model_name}/client/
        echo "convert ${model_name} format suucessfully"

        tar czf ${model_name}.tar.gz ${model_name}/
        echo "compress and tar ${model_name} suucessfully"
        """

        container = {
            "image": "registry.baidubce.com/paddleflow-public/serving:v0.6.2",
            "command": ["sh", "-exc", convert_shell],
            "args": [mount_path, model_name, dirname, pdmodel, pdiparams],
            "volumeMounts": [{
                "name": "task-center",
                "mountPath": "/mnt/task-center/",
            }],
        }

        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"{spec.get('name')}-serving-convert",
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [container],
                        "volumes": [{
                            "name": "task-center",
                            "persistentVolumeClaim": {
                                "claimName": spec.get("pvc_name"),
                            }
                        }]
                    }
                }
            }
        }

        return job


class Service:

    def __init__(self, client=None):
        self.client = k8s_client.CoreV1Api(client)

    def patch(self, spec):
        """Apply namespaced service
          Args:
            spec: The spec for the CR
        """
        name = spec["metadata"]["name"]
        namespace = spec["metadata"].get("namespace", "default")
        logger.info("Patching service %s in namespace %s.", name, namespace)
        api_response = self.client.patch_namespaced_service(name, namespace, spec)
        logger.info("Patched service %s in namespace %s.", name, namespace)
        return api_response

    def create(self, spec):
        """Create a CR.
        Args:
          spec: The spec for the CR.
        """
        # Create a Resource
        name = spec["metadata"]["name"]
        namespace = spec["metadata"].get("namespace", "default")
        logger.info("Creating service %s in namespace %s.", name, namespace)
        api_response = self.client.create_namespaced_service(namespace, spec)
        logger.info("Created service %s in namespace %s.", name, namespace)
        return api_response

    def apply(self, spec):
        """Create or update a CR
        Args:
          spec: The spec for the CR.
        """
        name = spec["metadata"]["name"]
        namespace = spec["metadata"].get("namespace", "default")
        logger.info("Apply service %s in namespace %s.", name, namespace)

        try:
            api_response = self.client.create_namespaced_service(namespace, spec)
            return api_response
        except rest.ApiException as err:
            if str(err.status) != "409":
                raise

        logger.info("Already exists now begin updating")
        api_response = self.client.patch_namespaced_service(name, namespace, spec)
        logger.info("Applied service %s in namespace %s.", name, namespace)
        return api_response

    def delete(self, name, namespace):
        logger.info("Deleting service %s in namespace %s.", name, namespace)
        api_response = self.client.delete_namespaced_service(name, namespace)
        logger.info("Deleted service %s in namespace %s.", name, namespace)
        return api_response

    def get_spec(self, spec):
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{spec.get('name')}-visualdl",
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "ports": [{
                    "name": "http",
                    "port": 8040,
                    "protocol": "TCP",
                    "targetPort": 8040
                }],
                "selector": {
                    "visualdl": f"{spec.get('name')}-visualdl"
                }
            }
        }

    def run(self, spec, action):
        inst_spec = self.get_spec(spec)
        print(f"The spec of crd is {inst_spec}")
        if action == "create":
            response = self.create(inst_spec)
        elif action == "patch":
            response = self.patch(inst_spec)
        elif action == "apply":
            response = self.apply(inst_spec)
        elif action == "delete":
            response = self.delete(spec.get("name"), spec.get("namespace"))
            print(f"Delete /v1/services have response {response}")
            return
        else:
            raise Exception("action must be one of create/patch/apply/delete")

        print(f"{action} /v1/services have response {response}")


def main():
    parser = argparse.ArgumentParser(description='PaddleJob launcher')
    parser.add_argument('--name', type=str,
                        help='The name of DataSet.')
    parser.add_argument('--namespace', type=str,
                        default='kubeflow',
                        help='The namespace of training task.')
    parser.add_argument('--action', type=str,
                        default='apply',
                        help='Action to execute on training task.')

    parser.add_argument('--project', type=str, required=True,
                        help='The project name of paddlepaddle ecosystem such as PaddleOCR')
    parser.add_argument('--image', type=str, required=True,
                        help='The image of paddle training job which contains model training scripts.')
    parser.add_argument('--config_path', type=str, required=True,
                        help='The path of model config, it is relative path from root path of project')
    parser.add_argument('--config_changes', type=str, required=True,
                        help='The key value pair of model config items, separate by comma, such as epoch=20.')
    parser.add_argument('--pretrain_model', type=str, default=None,
                        help='The uri of pretrained models where it store in.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The name of sample data set.')
    parser.add_argument('--train_label', type=str, default=None,
                        help='The name of train label file.')
    parser.add_argument('--test_label', type=str, default=None,
                        help='The name of test label file.')
    parser.add_argument('--pvc_name', type=str, required=True,
                        help='The persistent volume claim name of task-center.')
    parser.add_argument('--worker_replicas', type=int, required=True,
                        help='The replicas of worker pods.')
    parser.add_argument('--ps_replicas', type=int, default=0,
                        help='The replicas of parameter server pods.')
    parser.add_argument('--gpu_per_node', type=int, default=0,
                        help='Specified the number of gpu that training job requested.')
    parser.add_argument('--use_visualdl', type=strtobool, default=False,
                        help='Specified whether use VisualDL, this will be work only when worker replicas is 1.')
    parser.add_argument('--save_inference', type=str, default=None,
                        help='The command to convert training model to inference model.')
    # parser.add_argument('--need_convert', type=strtobool, default=True,
    #                     help='Convert model format so that it can be used by paddle serving.')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating DataSet template.')

    paddle_job_spec = {
        "name": args.name,
        "namespace": args.namespace,
        "project": args.project.lower(),
        "image": args.image,
        "config_path": args.config_path,
        "pretrain_model": args.pretrain_model,
        "dataset": args.sample_set,
        "train_label": args.train_label,
        "test_label": args.test_label,
        "worker_replicas": args.worker_replicas,
        "ps_replicas": args.ps_replicas,
        "gpu_per_node": args.gpu_per_node,
        "pvc_name": args.pvc_name,
    }

    visualdl_spec = {
        "name": args.name,
        "namespace": args.namespace,
        "pvc_name": args.pvc_name
    }

    service_spec = {
        "name": args.name,
        "namespace": args.namespace,
    }

    inference_spec = {
        "name": args.name,
        "namespace": args.namespace,
    }

    convert_spec = {
        "name": args.name,
        "namespace": args.namespace,
    }

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    paddle_job = PaddleJob(client=api_client)
    paddle_job.run(paddle_job_spec, action=args.action)

    if args.convert_command:
        inference_op = InferenceOp(client=api_client)
        inference_op.run(inference_spec, action=args.action)
        if args.action != "delete":
            inference_op.run(inference_spec, action="delete")

    if args.need_convert:
        convert_op = ConvertOp(client=api_client)
        convert_op.run(convert_spec, action=args.action)
        if args.action != "delete":
            convert_op.run(convert_spec, action="delete")

    if args.use_visualdl and args.worker_replicas == 1:
        visualDL = VisualDL(client=api_client)
        service = Service(client=api_client)
        visualDL.run(visualdl_spec, action=args.action)
        service.run(service_spec, action=args.action)

    if args.action != "delete":
        paddle_job.run(paddle_job_spec, action="delete")


if __name__ == "__main__":
    main()
