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


SampleJobGroup = "batch.paddlepaddle.org"
SampleJobPlural = "samplejobs"


class SampleJob(launch_crd.K8sCR):

    def __init__(self, version="v1alpha1", client=None):
        super(SampleJob, self).__init__(SampleJobGroup, SampleJobPlural, version, client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("phase")
        if not conditions:
            return False, ""
        if conditions in expected_conditions:
            return True, conditions
        else:
            return False, conditions


def main(argv=None):
    parser = argparse.ArgumentParser(description='SampleJob launcher')
    parser.add_argument('--name', type=str,
                        help='SampleJob name.')
    parser.add_argument('--namespace', type=str,
                        default='kubeflow',
                        help='SampleJob namespace.')
    parser.add_argument('--version', type=str,
                        default='v1alpha1',
                        help='SampleJob version.')
    parser.add_argument('--timeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for the SampleJob to reach end')
    parser.add_argument('--deleteAfterDone', type=strtobool,
                        default=False,
                        help='Delete SampleJob after the job is done')

    parser.add_argument('--type', type=str,
                        default="sync",
                        help='Job Type of SampleJob. One of the four types: `sync`, `warmup`, `rmr`, `clear`')
    parser.add_argument('--sampleSetRef', type=yamlOrJsonStr,
                        default={},
                        help='The information of reference SampleSet object.')
    parser.add_argument('--secretRef', type=yamlOrJsonStr,
                        default={},
                        help='Used for sync job, if the source data storage requires additional authorization information.')
    parser.add_argument('--terminate', type=strtobool,
                        default=False,
                        help='terminate other jobs that already in event queue of runtime servers')
    parser.add_argument('--jobOptions', type=yamlOrJsonStr,
                        default={},
                        help='Options for SampleJob.')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info('Generating SampleJob template.')

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    sample_job = SampleJob(version=args.version, client=api_client)
    inst = {
        "apiVersion": "%s/%s" % (SampleJobGroup, args.version),
        "kind": "SampleJob",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "type": args.type,
        },
    }
    if args.terminate:
        inst["spec"]["terminate"] = True
    else:
        inst["spec"]["terminate"] = False

    if args.sampleSetRef:
        inst["spec"]["sampleSetRef"] = args.sampleSetRef
    if args.secretRef:
        inst["spec"]["secretRef"] = args.secretRef
    if args.jobOptions:
        inst["spec"].update(args.jobOptions)

    create_response = sample_job.create(inst)
    print("create SampleJob have response {}".format(create_response))

    expected_conditions = ["Succeeded", "Failed"]
    sample_job.wait_for_condition(
        args.namespace, args.name, expected_conditions,
        timeout=datetime.timedelta(minutes=args.timeoutMinutes))
    if args.deleteAfterDone:
        sample_job.delete(args.name, args.namespace)


if __name__ == "__main__":
    main()
