"""
Python Script to run execute SageMaker Pipelines containing steps to perform the following:
- Preprocess a dataset using a Data Wrangler flow
- Train a model on the preprocessed dataset using SageMaker Autopilot
- Evaluate the model on a test dataset
- Register the evaluated model to the Model Registry

"""
import os
import re
import boto3
import json
import pandas as pd
import time
import uuid

import sagemaker
import sagemaker.session

from sagemaker import (
    AutoML,
    AutoMLInput,
    get_execution_role,
    MetricsSource,
    ModelMetrics,
    ModelPackage,
    image_uris,
)
from sagemaker.predictor import Predictor
from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.s3 import s3_path_join, S3Downloader, S3Uploader
from sagemaker.serializers import CSVSerializer
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import Processor
from sagemaker.transformer import Transformer
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep

from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TransformStep, TransformInput
from sagemaker.network import NetworkConfig

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AutopilotPackageGroup",
    pipeline_name="AutopilotPipeline",
    base_job_prefix="Autopilot",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on UCI adult data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """

    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = PipelineSession()
    sagemaker_client = boto3.client("sagemaker")
    output_prefix = "auto-ml-training"

    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

    max_automl_runtime = ParameterInteger(
        name="MaxAutoMLRuntime", default_value=3600
    )  # max. AutoML training runtime: 1 hour
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    model_registration_metric_threshold = ParameterFloat(
        name="ModelRegistrationMetricThreshold", default_value=0.7
    )
    s3_bucket = ParameterString(
        name="S3Bucket", default_value=pipeline_session.default_bucket()
    )
    target_attribute_name = ParameterString(
        name="TargetAttributeName", default_value="_c14"  # class
    )

    feature_names = [
        "_c0",  # age
        "_c1",  # workclass
        "_c2",  # fnlwgt
        "_c3",  # education
        "_c4",  # education-num
        "_c5",  # marital-status
        "_c6",  # occupation
        "_c7",  # relationship
        "_c8",  # race
        "_c9",  # sex
        "_c10",  # capital-gain
        "_c11",  # capital-loss
        "_c12",  # hours-per-week
        "_c13",  # native-country
    ]
    column_names = feature_names + [target_attribute_name.default_value]

    dataset_file_name = "adult.data"

    S3Downloader.download(
        f"s3://sagemaker-sample-files/datasets/tabular/uci_adult/{dataset_file_name}",
        ".",
        sagemaker_session=pipeline_session,
    )
    df = pd.read_csv(dataset_file_name, header=None, names=column_names)
    df.to_csv("train_val.csv", index=False)

    dataset_file_name = "adult.test"
    S3Downloader.download(
        f"s3://sagemaker-sample-files/datasets/tabular/uci_adult/{dataset_file_name}",
        ".",
        sagemaker_session=pipeline_session,
    )
    df = pd.read_csv(dataset_file_name, header=None, names=column_names, skiprows=1)
    df[target_attribute_name.default_value] = df[
        target_attribute_name.default_value
    ].map({" <=50K.": " <=50K", " >50K.": " >50K"})
    df.to_csv(
        "x_test.csv",
        header=False,
        index=False,
        columns=[
            x for x in column_names if x != target_attribute_name.default_value
        ],  # all columns except target
    )
    df.to_csv(
        "y_test.csv",
        header=False,
        index=False,
        columns=[target_attribute_name.default_value],
    )

    s3_prefix = s3_path_join("s3://", s3_bucket.default_value, "data")
    S3Uploader.upload("train_val.csv", s3_prefix, sagemaker_session=pipeline_session)
    S3Uploader.upload("x_test.csv", s3_prefix, sagemaker_session=pipeline_session)
    S3Uploader.upload("y_test.csv", s3_prefix, sagemaker_session=pipeline_session)
    s3_train_val = s3_path_join(s3_prefix, "train_val.csv")
    s3_x_test = s3_path_join(s3_prefix, "x_test.csv")
    s3_y_test = s3_path_join(s3_prefix, "y_test.csv")

    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-sample-files/datasets/tabular/uci_adult/adult.data",
    )
    refit_flow = ParameterBoolean(name="RefitFlow", default_value=False)

    flow_file_name = "preprocess.flow"

    flow_file = "pipelines/autopilot/preprocess.flow"

    with open(flow_file, "r") as f:
        flow = json.loads(f.read())

    # Upload flow to S3

    flow_prefix = s3_path_join("s3://", s3_bucket.default_value, "data-wrangler-flows")
    S3Uploader.upload(flow_file, flow_prefix, sagemaker_session=pipeline_session)

    flow_s3_uri = s3_path_join(flow_prefix, "preprocess.flow")

    s3_output_base_path = (
        f"s3://{s3_bucket.default_value}/auto-ml-training/dw_flow_output/"
    )
    processing_job_outputs, output_names = create_processing_job_outputs(
        flow, s3_output_base_path
    )

    flow_input = ProcessingInput(
        source=flow_s3_uri,
        destination="/opt/ml/processing/flow",
        input_name="flow",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
    )

    # Data Wrangler Container URI.
    container_uri = image_uris.retrieve(framework="data-wrangler", region=region)

    # Processing Job Instance count and instance type.
    dw_instance_count = 2
    dw_instance_type = "ml.m5.4xlarge"

    # Size in GB of the EBS volume to use for storing data during processing.
    volume_size_in_gb = 30

    # Content type for each output. Data Wrangler supports CSV as default and Parquet.
    output_content_type = "CSV"
    enable_network_isolation = False

    # List of tags to be passed to the processing job.
    user_tags = []

    output_configs = [
        {
            output_name[1]: {
                "content_type": output_content_type,
                # "delimiter": delimiter,
                # "compression": compression,
                # "partition_config": partition_config,
            }
        }
        for output_name in output_names
    ]

    parameter_overrides = {
        "InputDataUrl": input_data,
    }

    parameter_override_args = create_parameter_override_args(parameter_overrides)
    # KMS key for per object encryption; default is None.
    kms_key = None

    network_config = NetworkConfig(
        enable_network_isolation=enable_network_isolation,
        security_group_ids=None,
        subnets=None,
    )

    processor = Processor(
        role=role,
        image_uri=container_uri,
        instance_count=dw_instance_count,
        instance_type=dw_instance_type,
        volume_size_in_gb=volume_size_in_gb,
        network_config=network_config,
        sagemaker_session=pipeline_session,
        output_kms_key=kms_key,
        tags=user_tags,
    )

    data_wrangler_step = ProcessingStep(
        name="DataWranglerProcessingStep",
        processor=processor,
        inputs=[flow_input],
        outputs=processing_job_outputs,
        job_arguments=[
            f"--output-config '{json.dumps(output_config)}'"
            for output_config in output_configs
        ]
        + [
            Join(
                on="",
                values=[
                    f'--refit-trained-params \'{{"refit": ',
                    refit_flow,
                    f', "output_flow": "{flow_file_name}"}}\'',
                ],
            )
        ]
        + create_parameter_override_args(parameter_overrides),
    )

    # End Pipeline DW Flow processing

    dw_output_name = [
        output_name[1]
        for output_name in output_names
        if "adult-dw-processed.data" in output_name[0].lower()
    ][0]

    automl_input = Join(
        on="/",
        values=[
            data_wrangler_step.properties.ProcessingOutputConfig.Outputs[
                dw_output_name
            ].S3Output.S3Uri,
            data_wrangler_step.properties.ProcessingJobName,
        ],
    )

    target_attribute_name = "_c14"  # comment this if skipping Data Wrangler step

    automl = AutoML(
        role=role,
        target_attribute_name=target_attribute_name,
        sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=max_automl_runtime,
        mode="ENSEMBLING",  # only ensembling mode is supported for native AutoML step integration in SageMaker Pipelines
    )
    train_args = automl.fit(
        inputs=[
            AutoMLInput(
                # inputs=s3_train_val, #Use this if you are skipping Data Wrangler step
                inputs=automl_input,  # comment this if skipping Data Wrangler step
                target_attribute_name=target_attribute_name,
                channel_type="training",
            )
        ]
    )

    step_auto_ml_training = AutoMLStep(
        name="AutoMLTrainingStep",
        step_args=train_args,
    )

    best_auto_ml_model = step_auto_ml_training.get_best_auto_ml_model(
        role, sagemaker_session=pipeline_session
    )
    step_args_create_model = best_auto_ml_model.create(instance_type=instance_type)
    step_create_model = ModelStep(
        name="ModelCreationStep", step_args=step_args_create_model
    )

    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=Join(
            on="/", values=["s3:/", s3_bucket, output_prefix, "transform"]
        ),
        sagemaker_session=pipeline_session,
    )
    step_batch_transform = TransformStep(
        name="BatchTransformStep",
        step_args=transformer.transform(data=s3_x_test, content_type="text/csv"),
    )

    evaluation_report = PropertyFile(
        name="evaluation",
        output_name="evaluation_metrics",
        path="evaluation_metrics.json",
    )

    sklearn_processor = SKLearnProcessor(
        role=role,
        framework_version="1.0-1",
        instance_count=instance_count,
        instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
    )
    step_args_sklearn_processor = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/input/predictions",
            ),
            ProcessingInput(
                source=s3_y_test, destination="/opt/ml/processing/input/true_labels"
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source="/opt/ml/processing/evaluation",
                destination=Join(
                    on="/", values=["s3:/", s3_bucket, output_prefix, "evaluation"]
                ),
            ),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=step_args_sklearn_processor,
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_auto_ml_training.properties.BestCandidateProperties.ModelInsightsJsonReportPath,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=step_auto_ml_training.properties.BestCandidateProperties.ExplainabilityJsonReportPath,
            content_type="application/json",
        ),
    )
    step_args_register_model = best_auto_ml_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[instance_type],
        transform_instances=[instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register_model = ModelStep(
        name="ModelRegistrationStep", step_args=step_args_register_model
    )

    step_conditional_registration = ConditionStep(
        name="ConditionalRegistrationStep",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluation.name,
                    property_file=evaluation_report,
                    json_path="classification_metrics.weighted_f1.value",
                ),
                right=model_registration_metric_threshold,
            )
        ],
        if_steps=[step_register_model],
        else_steps=[],  # pipeline end
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_count,
            instance_type,
            dw_instance_count,
            dw_instance_type,
            max_automl_runtime,
            model_approval_status,
            model_package_group_name,
            model_registration_metric_threshold,
            s3_bucket,
            target_attribute_name,
            input_data,
            refit_flow,
        ],
        steps=[
            data_wrangler_step,
            step_auto_ml_training,
            step_create_model,
            step_batch_transform,
            step_evaluation,
            step_conditional_registration,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def filter_string(s):
    return "-".join(re.findall(r"[a-zA-Z0-9!.*'()_-]+", s))


def get_destination_node_output_names(flow):
    output_names = []
    for node in flow["nodes"]:
        if node["type"] == "DESTINATION":
            output_names.append(
                (
                    filter_string(node["name"]),
                    f"{node['node_id']}.{node['outputs'][0]['name']}",
                )
            )
    return output_names


def create_processing_job_outputs(flow, s3_output_base_path):
    output_names = get_destination_node_output_names(flow)
    processing_outputs = []
    for dataset_name, output_name in output_names:
        processing_outputs.append(
            ProcessingOutput(
                output_name=output_name,
                source=f"/opt/ml/processing/output/{dataset_name}",
                destination=os.path.join(s3_output_base_path, dataset_name),
                s3_upload_mode="EndOfJob",
            )
        )
    return processing_outputs, output_names


def create_parameter_override_args(parameter_overrides):
    """Create PJ args from parameter overrides.

    Args:
        parameter_overrides: a mapping of parameter name to Pipeline Parameter object
    Returns: list of `--parameter-override` container arguments
    """
    return [
        Join(on="", values=[f'--parameter-override \'{{"{name}": "', value, "\"}'"])
        for name, value in parameter_overrides.items()
    ]
