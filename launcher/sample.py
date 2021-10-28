import json
from typing import NamedTuple
from collections import namedtuple
import kfp
import kfp.dsl as dsl
from kfp import components
from kfp.dsl.types import Integer


SAMPLE_SET_IMAGE = ""
SAMPLE_JOB_IMAGE = ""
PADDLE_JOB_IMAGE = "registry.baidubce.com/paddle-operator/demo-wide-and-deep:v1"


@dsl.pipeline(
    name="launch-kubeflow-paddle",
    description="An example to launch paddle.",
)
def wide_deep_demo():
    sampleset_launcher_op = components.load_component_from_file("./sampleset-component.yaml")
    components.load_component_from_file("./sa")
    paddlejob_launcher_op = components.load_component_from_file("./paddlejob-component.yaml")

    worker = {
        "replicas": 1,
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "paddle",
                        "image": PADDLE_JOB_IMAGE,
                    },
                ]
            }
        }
    }

    paddlejob_launcher_op(
        name="wide-and-deep",
        namespace="kubeflow",
        worker_spec=worker,
        ps_spec=worker,
    )


if __name__ == "__main__":
    import kfp.compiler as compiler

    pipeline_file = "wide_deep_demo.yaml"

    compiler.Compiler().compile(wide_deep_demo, pipeline_file)

    client = kfp.Client(host="http://www.my-pipeline-ui.com:80")
    run = client.create_run_from_pipeline_package(
        pipeline_file,
        arguments={},
        run_name="paddle test run",
        service_account="paddle-operator"
    )
    print(f"Created run {run}")
