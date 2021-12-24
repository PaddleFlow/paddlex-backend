import argparse
import datetime
from distutils.util import strtobool
import logging
import yaml
import launch_crd

from kubernetes import client as k8s_client
from kubernetes import config


def yamlOrJsonStr(str):
    if str == "" or str == None:
        return None
    return yaml.safe_load(str)


class TaskCenterVolume(launch_crd.K8sCR):

    def __init__(self):
        pass

    def is_expected_conditions(self, inst, expected_conditions):
        pass


class SampleSet(launch_crd.K8sCR):

    def __init__(self, client=None):
        super(SampleSet, self).__init__("batch.paddlepaddle.org", "samplesets", "v1alpha1", client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("phase")
        if not conditions:
            return False, ""
        if conditions in expected_conditions:
            return True, conditions
        else:
            return False, conditions


class SampleJob(launch_crd.K8sCR):

    def __init__(self, client=None):
        super(SampleJob, self).__init__("batch.paddlepaddle.org", "samplejobs", "v1alpha1", client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("phase")
        if not conditions:
            return False, ""
        if conditions in expected_conditions:
            return True, conditions
        else:
            return False, conditions


class PaddleServing(launch_crd.K8sCR):

    def __init__(self, client=None):
        super(PaddleServing, self).__init__("elasticserving.paddlepaddle.org", "paddleservices", "v1", client)


class KnativeService(launch_crd.K8sCR):

    def __init__(self, client=None):
        super(KnativeService, self).__init__("serving.knative.dev", "services", "v1", client)

    def get_service(self, namespace, name):
        try:
            ksvc = self.client.get_namespaced_custom_object(
                self.group, self.version, namespace, self.plural, name)
        except Exception as e:
            raise Exception("There was a problem waiting for %s/%s %s in namespace %s; Exception: %s",
                            self.group, self.plural, name, namespace, e)
        return ksvc


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

    def run(self, name, namespace, ):
        inst = {
            "apiVersion": "%s/%s" % (self.group, self.version),
            "kind": "PaddleJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": {
                "cleanPodPolicy": "OnCompletion",
                "withGloo": 1,
                "intranet": "PodIP",
            },
        }





def main(argv=None):
    parser = argparse.ArgumentParser(description='PaddleJob launcher')
    parser.add_argument('--name', type=str,
                        help='PaddleJob name.')
    parser.add_argument('--namespace', type=str,
                        default='kubeflow',
                        help='PaddleJob namespace.')
    parser.add_argument('--version', type=str,
                        default='v1',
                        help='PaddleJob version.')
    parser.add_argument('--action', type=str,
                        default='create',
                        help='Action to execute on PaddleJob.')
    parser.add_argument('--timeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for the PaddleJob to reach end')
    parser.add_argument('--deleteAfterDone', type=strtobool,
                        default=False,
                        help='delete PaddleJob after the job is done')

    parser.add_argument('--cleanPodPolicy', type=str,
                        default="OnCompletion",
                        help='defines whether to clean pod after job finished')
    parser.add_argument('--schedulingPolicy', type=yamlOrJsonStr,
                        default={},
                        help='defines the policy related to scheduling, for volcano')
    parser.add_argument('--intranet', type=str,
                        default="PodIP",
                        help='defines the communication mode inter pods : PodIP, Service or Host')
    parser.add_argument('--withGloo', type=int,
                        default=1,
                        help='indicate whether enable gloo, 0/1/2 for disable/enable for worker/enable for server')
    parser.add_argument('--sampleSetRef', type=yamlOrJsonStr,
                        default={},
                        help='defines the sample data set used for training and its mount path in worker pods')
    parser.add_argument('--ps', type=yamlOrJsonStr,
                        default={},
                        help='describes the spec of server base on pod template')
    parser.add_argument('--worker', type=yamlOrJsonStr,
                        default={},
                        help='describes the spec of worker base on pod template')
    parser.add_argument('--heter', type=yamlOrJsonStr,
                        default={},
                        help='describes the spec of heter worker base on pod temlate')
    parser.add_argument('--elastic', type=int,
                        default=0,
                        help='indicate the elastic level')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating PaddleJob template.')

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    pdj = PaddleJob(version=args.version, client=api_client)
    inst = {
        "apiVersion": "%s/%s" % (PaddleJobGroup, args.version),
        "kind": "PaddleJob",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "cleanPodPolicy": args.cleanPodPolicy,
            "withGloo": args.withGloo,
            "intranet": args.intranet,
            "worker": {

            },
        },
    }

    if args.schedulingPolicy:
        inst["spec"]["schedulingPolicy"] = args.schedulingPolicy
    if args.sampleSetRef:
        inst["spec"]["sampleSetRef"] = args.sampleSetRef
    if args.ps:
        inst["spec"]["ps"] = args.ps
    if args.worker:
        inst["spec"]["worker"] = args.worker
    if args.heter:
        inst["spec"]["heter"] = args.heter
    if args.elastic > 0:
        inst["spec"]["elastic"] = args.elastic

    if args.action == "create":
        response = pdj.create(inst)
    elif args.action == "patch":
        response = pdj.patch(inst)
    elif args.action == "apply":
        response = pdj.apply(inst)
    elif args.action == "delete":
        response = pdj.delete(args.name, args.namespace)
        print("Delete PaddleJob have response {}".format(response))
        return
    else:
        raise Exception("action must be one of create/patch/apply/delete")

    print("{} PaddleJob have response {}".format(args.action, response))

    expected_conditions = ["Succeed", "Completed"]
    error_phases = ["Failed", "Terminated"]
    pdj.wait_for_condition(
        args.namespace, args.name, expected_conditions, error_phases,
        timeout=datetime.timedelta(minutes=args.timeoutMinutes))

    if args.deleteAfterDone:
        pdj.delete(args.name, args.namespace)


if __name__ == "__main__":
    main()
