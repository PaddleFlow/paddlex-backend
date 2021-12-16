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
    def __init__(self, version="v1", client=None):
        super(VisualDL, self).__init__("apps", "deployments", version, client)

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
                        "serviceAccountName": "pipeline-runner",
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
                        default='create',
                        help='Action to execute on training task.')

    parser.add_argument('--image', type=str,
                        help='The image of paddle training job which contains model training scripts.')
    parser.add_argument('--command', type=str,
                        help='The command to start model training task.')
    parser.add_argument('--dataset', type=str,
                        help='The name of sample data set.')
    parser.add_argument('--pvc_name', type=str,
                        help='The persistent volume claim name of task-center.')
    parser.add_argument('--worker_replicas', type=int,
                        help='The replicas of worker pods.')
    parser.add_argument('--ps_replicas', type=int, default=0,
                        help='The replicas of parameter server pods.')
    parser.add_argument('--gpu_per_node', type=int, default=0,
                        help='Specified the number of gpu that training job requested.')
    parser.add_argument('--use_visualdl', type=strtobool, default=False,
                        help='Specified whether use VisualDL, this will be work only when worker replicas is 1')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating DataSet template.')

    paddle_job_spec = {
        "name": args.name,
        "namespace": args.namespace,
        "image": args.image,
        "command": args.command,
        "dataset": args.sample_set,
        "worker_replicas": args.worker_replicas,
        "ps_replicas": args.ps_replicas,
        "gpu_per_node": args.gpu_per_node,
        "pvc_name": args.pvc_name
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

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    paddle_job = PaddleJob(client=api_client)
    paddle_job.run(paddle_job_spec, action=args.action)

    if args.use_visualdl and args.worker_replicas == 1:
        visualDL = VisualDL(client=api_client)
        service = Service(client=api_client)
        visualDL.run(visualdl_spec, action=args.action)
        service.run(service_spec, action=args.action)

    if args.action != "delete":
        paddle_job.run(paddle_job_spec, action="delete")


if __name__ == "__main__":
    main()
