import kfp.dsl as dsl
import datetime
import os
from kubernetes import client as k8s_client

@dsl.pipeline(
    name='wide_deep_critio_pipeline',
    description='Demonstrate an end-to-end training & serving pipeline using Wide & Deep and Critio'
)
def resnet_pipeline(
    raw_data_dir='/mnt/workspace/raw_data',
    processed_data_dir='/mnt/workspace/processed_data',
    model_dir='/mnt/workspace/saved_model',
    epochs=50,
    trtserver_name='trtis',
    model_name='resnet_graphdef',
    model_version=1,
    webapp_prefix='webapp',
    webapp_port=80
):

    persistent_volume_name = 'nvidia-workspace'
    persistent_volume_path = '/mnt/workspace'

    op_dict = {}

    op_dict['preprocess'] = PreprocessOp(
        'preprocess', raw_data_dir, processed_data_dir)

    op_dict['train'] = TrainOp(
        'train', op_dict['preprocess'].output, model_dir, model_name, model_version, epochs)

    op_dict['deploy_inference_server'] = InferenceServerLauncherOp(
        'deploy_inference_server', op_dict['train'].output, trtserver_name)

    op_dict['deploy_webapp'] = WebappLauncherOp(
        'deploy_webapp', op_dict['deploy_inference_server'].output, model_name, model_version, webapp_prefix, webapp_port)

    for _, container_op in op_dict.items():
        container_op.add_volume(k8s_client.V1Volume(
            host_path=k8s_client.V1HostPathVolumeSource(
                path=persistent_volume_path),
            name=persistent_volume_name))
        container_op.add_volume_mount(k8s_client.V1VolumeMount(
            mount_path=persistent_volume_path,
            name=persistent_volume_name))


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(resnet_pipeline, __file__ + '.tar.gz')
