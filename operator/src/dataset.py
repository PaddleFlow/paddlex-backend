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

    def get_spec(self, spec):
        return {
            "apiVersion": "%s/%s" % (self.group, self.version),
            "kind": "SampleSet",
            "metadata": {
                "name": spec.get("name"),
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "partitions": spec.get("partitions"),
                "source": {
                    "uri": spec.get("source_uri"),
                    "secretRef": spec.get("secret_name"),
                }
            },
        }


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


class SyncSampleJob(SampleJob):

    def get_spec(self, spec):
        return {
            "apiVersion": "%s/%s" % (self.group, self.version),
            "kind": "SampleJob",
            "metadata": {
                "name": f"{spec.get('name')}-sync",
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "type": "sync",
                "sampleSetRef": {
                    "name": spec.get("name")
                },
                "syncOptions": {
                    "source": spec.get("source_uri"),
                }
            },
        }


class WarmupSampleJob(SampleJob):

    def get_spec(self, spec):
        return {
            "apiVersion": "%s/%s" % (self.group, self.version),
            "kind": "SampleJob",
            "metadata": {
                "name": f"{spec.get('name')}-warmup",
                "namespace": spec.get("namespace"),
            },
            "spec": {
                "type": "warmup",
                "sampleSetRef": {
                    "name": spec.get("name")
                },
            },
        }


def main():
    parser = argparse.ArgumentParser(description='PaddleJob launcher')
    parser.add_argument('--name', type=str,
                        help='The name of DataSet.')
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
                        default=60 * 24,
                        help='Time in minutes to wait for the PaddleJob to reach end')
    parser.add_argument('--deleteAfterDone', type=strtobool,
                        default=False,
                        help='delete PaddleJob after the job is done')

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
