"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import os
from pathlib import Path
from aws_cdk import (
    CfnCondition,
    Fn,
    aws_lambda as lambda_,
    aws_apigateway as api_gw,
    aws_efs as efs,
    aws_ec2 as ec2
)
from aws_cdk import App, Stack, Duration, RemovalPolicy
from constructs import Construct

class ServerlessHuggingFaceStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # EFS needs to be setup in a VPC (but no NAT gateways, please)
        # Source: https://github.com/aws/aws-cdk/issues/1305
        vpc = ec2.Vpc(self, 'Vpc', max_azs=2, nat_gateways=0)
        exclude_condition = CfnCondition(
            self, 'exclude-default-route-subnet', expression=Fn.condition_equals(True, False))
        for subnet in vpc.private_subnets:
            for child in subnet.node.children:
                if type(child) == ec2.CfnRoute:
                    route: ec2.CfnRoute = child
                    route.cfn_options.condition = exclude_condition  # key point here
        # vpc = ec2.Vpc(self, 'Vpc', max_azs=2)

        # creates a file system in EFS to store cache models
        fs = efs.FileSystem(self, 'FileSystem',
                            vpc=vpc,
                            removal_policy=RemovalPolicy.DESTROY)
        access_point = fs.add_access_point('MLAccessPoint',
                                           create_acl=efs.Acl(
                                               owner_gid='1001', owner_uid='1001', permissions='750'),
                                           path="/export/models",
                                           posix_user=efs.PosixUser(gid="1001", uid="1001"))

        # %%
        # iterates through the Python files in the docker directory
        docker_folder = os.path.dirname(
            os.path.realpath(__file__)) + "/inference"
        pathlist = Path(docker_folder).rglob('*.py')
        for path in pathlist:
            base = os.path.basename(path)
            filename = os.path.splitext(base)[0]
            # Lambda Function from docker image
            function = lambda_.DockerImageFunction(
                self, filename,
                code=lambda_.DockerImageCode.from_image_asset(docker_folder,
                                                              cmd=[
                                                                  filename+".handler"]
                                                              ),
                memory_size=8096,
                timeout=Duration.seconds(600),
                vpc=vpc,
                filesystem=lambda_.FileSystem.from_efs_access_point(
                    access_point, '/mnt/hf_models_cache'),
                environment={
                    "TRANSFORMERS_CACHE": "/mnt/hf_models_cache",
                    "MODEL_DIR": "model",
                    "MODEL_FILENAME": "pytorch_model.bin",
                    "INCIDENTS_FILENAME": "incident_cls.pt",
                    "CSV_FILENAME": "incidents.csv",
                    "HF_MODEL_URI": "allenai/longformer-base-4096"
                },
            )

            # adds method for the function
            lambda_integration = api_gw.LambdaIntegration(function, proxy=False, integration_responses=[
                api_gw.IntegrationResponse(status_code='200',
                                           response_parameters={
                                               'method.response.header.Access-Control-Allow-Origin': "'*'"
                                           })
            ])


app = App()

ServerlessHuggingFaceStack(app, "ServerlessHuggingFaceStack")

app.synth()
# %%
