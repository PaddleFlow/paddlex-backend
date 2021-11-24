import argparse
import datetime
import logging
import yaml

from launch_crd import K8sCR, logger

from kubernetes import client as k8s_client
from kubernetes.client import rest
from kubernetes import config


def yamlOrJsonStr(str):
    if str == "" or str == None:
        return None
    return yaml.safe_load(str)


class VisualDL(K8sCR):
    def __init__(self, version="v1", client=None):
        super(VisualDL, self).__init__("apps", "deployments", version, client)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Serving launcher')
    parser.add_argument('--name', type=str,
                        help='PaddleService name.')
    parser.add_argument('--namespace', type=str,
                        default='kubeflow',
                        help='The namespace of PaddleService.')
    parser.add_argument('--timeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for deployment of VisualDL is ready.')
    parser.add_argument('--action', type=str,
                        default="apply",
                        help='Action to execute on deployment and service.')

    parser.add_argument('--pvc_name', type=str,
                        required=True,
                        help='The pvc claim name of pipeline.')
    parser.add_argument('--mount_path', type=str,
                        default="/mnt/task-center/",
                        help='The path that should mount to.')

    parser.add_argument('--logdir', type=str,
                        required=True,
                        help='Set one or more directories of the log.')
    parser.add_argument('--model', type=str,
                        help='Set a path to the model file (not a directory).')
    parser.add_argument('--port', type=int,
                        default=8040,
                        help='Set the port.')
    parser.add_argument('--cache_timeout', type=int,
                        help='Cache time of the backend. During the cache time, the front end requests the same URL '
                             'multiple times, and then the returned data are obtained from the cache.')
    parser.add_argument('--language', type=str,
                        help='The language of the VisualDL panel.')
    parser.add_argument('--public_path', type=str,
                        help='The URL path of the VisualDL panel.')
    parser.add_argument('--api_only', type=str,
                        help='Decide whether or not to provide only API.')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating VisualDL Deployment template.')

    install = "python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple && "

    cmd = f"visualdl --logdir {args.logdir} "
    if args.model:
        cmd += f"--model {args.model} "
    if args.port:
        cmd += f"--port {args.port} "
    if args.cache_timeout:
        cmd += f"--cache_timeout {args.cache_timeout} "
    if args.language:
        cmd += f"--language {args.language} "
    if args.public_path:
        cmd += f"--public_path {args.public_path} "
    if args.api_only:
        cmd += f"--api_only {args.api_only} "

    cmd = install + cmd

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "selector": {
                "matchLabels": {
                    "visualdl": args.name,
                }
            },
            "replicas": 1,
            "template": {
                "metadata": {
                    "labels": {
                        "visualdl": args.name,
                    },
                },
                "spec": {
                    "serviceAccountName": "pipeline-runner",
                    "containers": [{
                        "name": "visualdl",
                        "image": "python:3.7",
                        "imagePullPolicy": "IfNotPresent",
                        "workingDir": args.mount_path,
                        "volumeMounts": [{
                            "name": "task-center",
                            "mountPath": args.mount_path,
                        }],
                        "command": ["/bin/bash"],
                        "args": ["-c", cmd],
                        "ports": [{
                            "name": "http",
                            "containerPort": args.port
                        }]
                    }],
                    "volumes": [{
                        "name": "task-center",
                        "persistentVolumeClaim": {
                            "claimName": args.pvc_name,
                        }
                    }]
                }
            },
        },
    }

    service_inst = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "ports": [{
                "name": "http",
                "port": args.port,
                "protocol": "TCP",
                "targetPort": args.port
            }],
            "selector": {
                "visualdl": args.name
            }
        }
    }

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    visualdl = VisualDL(client=api_client)
    service = Service(client=api_client)

    if args.action == "create":
        response = visualdl.create(deployment)
        _ = service.create(service_inst)
    elif args.action == "patch":
        response = visualdl.patch(deployment)
        _ = service.patch(service_inst)
    elif args.action == "apply":
        response = visualdl.apply(deployment)
        _ = service.apply(service_inst)
    elif args.action == "delete":
        response = visualdl.delete(args.name, args.namespace)
        _ = service.delete(service_inst)
        print("Delete PaddleService have response {}".format(response))
        return
    else:
        raise Exception("action must be one of create/update/apply/delete")

    print("{} VisualDL deployment have response {}".format(args.action, response))

    expected_conditions = ["Available", "Progressing"]
    visualdl.wait_for_condition(
        args.namespace, args.name, expected_conditions, wait_created=True,
        timeout=datetime.timedelta(minutes=args.timeoutMinutes))


if __name__ == "__main__":
    main()
