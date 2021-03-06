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


SampleSetGroup = "batch.paddlepaddle.org"
SampleSetPlural = "samplesets"


class SampleSet(launch_crd.K8sCR):

    def __init__(self, version="v1alpha1", client=None):
        super(SampleSet, self).__init__(SampleSetGroup, SampleSetPlural, version, client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("phase")
        if not conditions:
            return False, ""
        if conditions in expected_conditions:
            return True, conditions
        else:
            return False, conditions


def main(argv=None):
    parser = argparse.ArgumentParser(description='SampleSet launcher')
    parser.add_argument('--name', type=str,
                        help='SampleSet name.')
    parser.add_argument('--namespace', type=str,
                        default='kubeflow',
                        help='SampleSet namespace.')
    parser.add_argument('--version', type=str,
                        default='v1alpha1',
                        help='SampleSet version.')
    parser.add_argument('--action', type=str,
                        default='apply',
                        help='Action to execute on SampleSet.')
    parser.add_argument('--timeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for the SampleSet to be ready.')

    parser.add_argument('--partitions', type=int,
                        default=1,
                        help='Partitions is the number of SampleSet partitions, partition means cache node.')
    parser.add_argument('--source', type=yamlOrJsonStr,
                        default={},
                        help='Source describes the information of data source uri and secret name.')
    parser.add_argument('--secretRef', type=yamlOrJsonStr,
                        default={},
                        help='SecretRef is reference to the authentication secret for source storage and cache engine.')
    parser.add_argument('--noSync', type=strtobool,
                        default=True,
                        help='If the data is already in cache engine backend storage, can set NoSync as true to skip Syncing phase.')
    parser.add_argument('--csi', type=yamlOrJsonStr,
                        default={},
                        help='CSI describes csi driver name and mount options to support cache data.')
    parser.add_argument('--cache', type=yamlOrJsonStr,
                        default={},
                        help='Cache options used by cache runtime engine.')
    parser.add_argument('--nodeAffinity', type=yamlOrJsonStr,
                        default={},
                        help='NodeAffinity defines constraints that limit what nodes this SampleSet can be cached to.')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating SampleSet template.')

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    sample_set = SampleSet(version=args.version, client=api_client)
    inst = {
        "apiVersion": "%s/%s" % (SampleSetGroup, args.version),
        "kind": "SampleSet",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "partitions": args.partitions,
        },
    }
    if args.noSync:
        inst["spec"]["noSync"] = True
    else:
        inst["spec"]["noSync"] = False

    if args.source:
        inst["spec"]["source"] = args.source
    if args.secretRef:
        inst["spec"]["secretRef"] = args.secretRef
    if args.csi:
        inst["spec"]["csi"] = args.csi
    if args.cache:
        inst["spec"]["cache"] = args.cache
    if args.nodeAffinity:
        inst["spec"]["nodeAffinity"] = args.nodeAffinity

    if args.action == "create":
        response = sample_set.create(inst)
    elif args.action == "patch":
        response = sample_set.patch(inst)
    elif args.action == "apply":
        response = sample_set.apply(inst)
    elif args.action == "delete":
        response = sample_set.delete(args.name, args.namespace)
        print("Delete SampleSet have response {}".format(response))
        return
    else:
        raise Exception("action must be one of create/patch/apply/delete")

    print("{} SampleSet have response {}".format(args.action, response))

    expected_conditions = ["Ready"]
    error_phases = ["SyncFailed", "PartialReady"]
    sample_set.wait_for_condition(
        args.namespace, args.name, expected_conditions, error_phases,
        timeout=datetime.timedelta(minutes=args.timeoutMinutes))


if __name__ == "__main__":
    main()
