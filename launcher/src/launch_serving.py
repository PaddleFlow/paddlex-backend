import argparse
import datetime
import os
import json
import logging
import yaml
import launch_crd

from kubernetes import client as k8s_client
from kubernetes import config


def yamlOrJsonStr(str):
    if str == "" or str == None:
        return None
    return yaml.safe_load(str)


PaddleServingGroup = "elasticserving.paddlepaddle.org"
PaddleServingPlural = "paddleservices"

KnativeServiceGroup = "serving.knative.dev"
KnativeServicePlural = "services"


class PaddleServing(launch_crd.K8sCR):
    def __init__(self, version="v1", client=None):
        super(PaddleServing, self).__init__(PaddleServingGroup, PaddleServingPlural, version, client)


class KnativeService(launch_crd.K8sCR):
    def __init__(self, version="v1", client=None):
        super(KnativeService, self).__init__(KnativeServiceGroup, KnativeServicePlural, version, client)

    def get_service(self, namespace, name):
        try:
            ksvc = self.client.get_namespaced_custom_object(
                self.group, self.version, namespace, self.plural, name)
        except Exception as e:
            raise Exception("There was a problem waiting for %s/%s %s in namespace %s; Exception: %s",
                            self.group, self.plural, name, namespace, e)
        return ksvc


def main(argv=None):
    parser = argparse.ArgumentParser(description='Serving launcher')
    parser.add_argument('--name', type=str,
                        help='PaddleService name.')
    parser.add_argument('--namespace', type=str,
                        default='kubeflow',
                        help='The namespace of PaddleService.')
    parser.add_argument('--version', type=str,
                        default='v1',
                        help='PaddleService version.')
    parser.add_argument('--timeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for the PaddleJob to reach end')
    parser.add_argument('--action', type=str,
                        default="create",
                        help='Action to execute on ElasticServing.')

    parser.add_argument('--runtimeVersion', type=str,
                        default="paddleserving",
                        help='Version of the service.')
    parser.add_argument('--resources', type=yamlOrJsonStr,
                        default={},
                        help='Defaults to requests and limits of 1CPU, 2Gb MEM.')
    parser.add_argument('--default', type=yamlOrJsonStr,
                        default={},
                        help='DefaultTag defines default PaddleService endpoints.')
    parser.add_argument('--canary', type=yamlOrJsonStr,
                        default={},
                        help='CanaryTag defines an alternative PaddleService endpoints.')
    parser.add_argument('--canaryTrafficPercent', type=yamlOrJsonStr,
                        default=0,
                        help='CanaryTrafficPercent defines the percentage of traffic going to canary PaddleService endpoints.')
    parser.add_argument('--service', type=yamlOrJsonStr,
                        default={},
                        help='Service defines the configuration for Knative Service.')
    parser.add_argument('--workingDir', type=str,
                        default={},
                        help='Working directory of container.')
    parser.add_argument('--volumeMounts', type=yamlOrJsonStr,
                        default={},
                        help='Pod volumes to mount into the container filesystem.')
    parser.add_argument('--volumes', type=yamlOrJsonStr,
                        default={},
                        help='List of volumes that can be mounted by containers belonging to the pod.')
    parser.add_argument('--outputPath', type=str,
                        default="",
                        help="The path store json format of knative service")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating PaddleService template.')

    inst = {
        "apiVersion": "%s/%s" % (PaddleServingGroup, args.version),
        "kind": "PaddleService",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "default": args.default,
            "runtimeVersion": args.runtimeVersion,
        },
    }

    if args.resources:
        inst["spec"]["resources"] = args.resources
    if args.canary:
        inst["spec"]["canary"] = args.canary
    if args.canaryTrafficPercent:
        inst["spec"]["canaryTrafficPercent"] = args.canaryTrafficPercent
    if args.service:
        inst["spec"]["service"] = args.service
    if args.workingDir:
        inst["spec"]["workingDir"] = args.workingDir
    if args.volumeMounts:
        inst["spec"]["volumeMounts"] = args.volumeMounts
    if args.volumes:
        inst["spec"]["volumes"] = args.volumes

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()

    serving = PaddleServing(version=args.version, client=api_client)
    if args.action == "create":
        response = serving.create(inst)
    elif args.action == "update":
        response = serving.update(inst)
    elif args.action == "delete":
        response = serving.delete(args.name, args.namespace)
    else:
        raise Exception("action must be one of create/update/delete")
    print("{} PaddleService have response {}".format(args.action, response))

    kservice = KnativeService(client=api_client)
    expected_conditions = ["RoutesReady", "Ready"]
    kservice.wait_for_condition(
        args.namespace, args.name, expected_conditions,
        timeout=datetime.timedelta(minutes=args.timeoutMinutes))

    if args.outputPath:
        ksvc = kservice.get_service(args.namespace, args.name)

        if not os.path.exists(os.path.dirname(args.outputPath)):
            os.makedirs(os.path.dirname(args.outputPath))
            with open(args.outputPath, "w") as report:
                report.write(json.dumps(ksvc))


if __name__ == "__main__":
    main()
