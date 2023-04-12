import argparse
import json
import logging
import os
import csv

import boto3
from botocore.exceptions import ClientError

from sagemaker.s3 import S3Downloader

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def invoke_endpoint(endpoint_name):
    boto3_session = boto3.Session()

    runtime = boto3_session.client("sagemaker-runtime")

    dataset_file_name = "adult.test"
    S3Downloader.download(
        f"s3://sagemaker-sample-files/datasets/tabular/uci_adult/{dataset_file_name}",
        ".",
    )

    with open("adult.test", "r") as file:
        csvreader = csv.reader(file)

        next(csvreader)

        output_dict = {}
        test_case_no = 0

        for row in csvreader:
            test_case_no += 1
            test_case_id = "test" + str(test_case_no)
            result_dict = {}
            endpoint_body = ""
            key = 0
            while key < 14:
                comma_value = ""
                if key < 13:
                    comma_value = ","

                endpoint_body = endpoint_body + row[key] + comma_value
                key += 1

            result_dict["input_data"] = endpoint_body

            # Send CSV text via InvokeEndpoint API
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="text/csv", Body=endpoint_body
            )
            resp_body = response["Body"]
            result = resp_body.read().decode("utf-8")
            result_list = result.split(",")
            result_dict["predicted_result"] = result_list[0].strip()
            result_dict["actual_result"] = row[14].strip().replace(".", "")
            if result_dict["predicted_result"] == result_dict["actual_result"]:
                result_dict["test_case"] = "passed"
            else:
                result_dict["test_case"] = "failed"

            output_dict[test_case_id] = result_dict

            if test_case_no == 100:
                break

    return output_dict


def test_endpoint(endpoint_name):
    """
    Describe the endpoint and ensure InSerivce, then invoke endpoint.  Raises exception on error.
    """
    error_message = None
    try:
        # Ensure endpoint is in service
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status != "InService":
            error_message = (
                f"SageMaker endpoint: {endpoint_name} status: {status} not InService"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Output if endpoint has data capture enbaled
        endpoint_config_name = response["EndpointConfigName"]
        response = sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        if (
            "DataCaptureConfig" in response
            and response["DataCaptureConfig"]["EnableCapture"]
        ):
            logger.info(
                f"data capture enabled for endpoint config {endpoint_config_name}"
            )

        # Call endpoint to handle
        return invoke_endpoint(endpoint_name)
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper()
    )
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # Load the build config
    with open(args.import_build_config, "r") as f:
        config = json.load(f)

    # Get the endpoint name from sagemaker project name
    endpoint_name = "{}-{}".format(
        config["Parameters"]["SageMakerProjectName"], config["Parameters"]["StageName"]
    )
    results = test_endpoint(endpoint_name)

    # Print results and write to file
    logger.debug(json.dumps(results, indent=4))
    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
