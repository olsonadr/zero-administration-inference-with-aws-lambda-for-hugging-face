"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

# Imports
import os
from dotenv import load_dotenv
from pathlib import Path
from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigatewayv2_alpha as api_gw2_a,
    aws_efs as efs,
    aws_ec2 as ec2
    # aws_sam as sam
    # aws_apigatewayv2 as api_gw2,
    # aws_apigateway as api_gw,
)
from aws_cdk import (App, Stack, Duration, RemovalPolicy,
                     Environment, CfnCondition, Fn, CfnOutput)
from aws_cdk.aws_apigatewayv2_integrations_alpha import HttpLambdaIntegration
from constructs import Construct

# Read dotenv for environment variables
load_dotenv()

# Region and account for stack
aws_env = Environment(
    account=os.environ['AWS_ACCOUNT_ID'], region=os.environ['AWS_REGION'])

# Stack for this cdk program
class ServerlessHuggingFaceStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # %%
        # EFS needs to be setup in a VPC (but no NAT gateways, maybe?)
        # Alternative for no nat gateway
        # Source: https://github.com/aws/aws-cdk/issues/1305#issuecomment-547361989
        vpc = ec2.Vpc(self, 'FargateVPC',
                      cidr='10.0.0.0/16',
                      max_azs=1,
                      nat_gateways=0,
                      subnet_configuration=[
                          {
                              'cidr_mask': 24,
                              'name': 'Public',
                              'subnetType': ec2.SubnetType.PUBLIC
                          }
                      ]
                      )

        # %%
        # Creates a file system in EFS to store cache models
        fs = efs.FileSystem(self, 'FileSystem',
                            vpc=vpc,
                            removal_policy=RemovalPolicy.DESTROY)
        access_point = fs.add_access_point('MLAccessPoint',
                                           create_acl=efs.Acl(
                                               owner_gid='1001', owner_uid='1001', permissions='750'),
                                           path="/export/models",
                                           posix_user=efs.PosixUser(gid="1001", uid="1001"))

        # %%
        # Create Http Gateway API
        # Source (adapted): https://bobbyhadz.com/blog/aws-cdk-api-gateway-example
        # Source (api for alpha): https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_apigatewayv2_alpha/HttpApi.html
        methods = [api_gw2_a.CorsHttpMethod.GET,
                   api_gw2_a.CorsHttpMethod.POST]
        httpApi = api_gw2_a.HttpApi(self, 'HttpApi',
                                  description='HTTP API Gateway for AIID Lambdas',
                                  cors_preflight={
                                      'allow_headers': ['Content-Type',
                                                       'X-Amz-Date',
                                                       'Authorization',
                                                       'X-Api-Key', ],
                                      'allow_methods': methods,
                                      'allow_credentials': True,
                                      'allow_origins': ['http://localhost:3000',
                                                        'http://www.apirequest.io/',
                                                        'https://www.apirequest.io/'],
                                  }
                                  )

        # Create output for the api url
        # this outputs the apiUrl to a json file when deploying the stack, so it can be used by programs as needed
        CfnOutput(self, 'apiUrl', value=httpApi.url)

        # %%
        # Iterates through the Python files in the docker directory
        docker_folder = os.path.dirname(
            os.path.realpath(__file__)) + "/inference"
        pathlist = Path(docker_folder).rglob('*.py')
        for path in pathlist:
            # Parse lambda handler filename name
            base = os.path.basename(path)
            filename = os.path.splitext(base)[0]

            # Lambda Function from docker image
            function = lambda_.DockerImageFunction(
                self, filename,
                allow_public_subnet=True,
                code=lambda_.DockerImageCode.from_image_asset(
                    docker_folder, cmd=[filename+".handler"]),
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

            # Create lambda integration for use in the Gateway API
            lambda_integration = HttpLambdaIntegration('LambdaIntegration', function)
            # use_proxy = True
            # lambda_integration = api_gw.LambdaIntegration(function, proxy=use_proxy, integration_responses=[
            #     api_gw.IntegrationResponse(status_code='200',
            #                                response_parameters={
            #                                    'method.response.header.Access-Control-Allow-Origin': "'*'"
            #                                })
            # ])

            # Add route lambda integration to a route on the api, using the fn name
            httpApi.add_routes(
                path=f"/{filename}",
                methods=[api_gw2_a.HttpMethod.GET, api_gw2_a.HttpMethod.POST],
                integration=lambda_integration
            )


# %%
# Initialize the aws stack
app = App()
ServerlessHuggingFaceStack(app, "ServerlessHuggingFaceStack", env=aws_env)
app.synth()


# # # # Backup vpc configurations
# # Custom, no nat gateway? (doesnt work yet)
# # Source: https://github.com/aws/aws-cdk/issues/1305
# vpc = ec2.Vpc(self, 'Vpc', max_azs=2, nat_gateways=0)
# exclude_condition = CfnCondition(
#     self, 'exclude-default-route-subnet', expression=Fn.condition_equals(True, False))
# for subnet in vpc.private_subnets:
#     for child in subnet.node.children:
#         if type(child) == ec2.CfnRoute:
#             route: ec2.CfnRoute = child
#             route.cfn_options.condition = exclude_condition  # key point here

# # Alternative for NAT instance rather than NAT gateway
# # Source: https://github.com/aws/aws-cdk/issues/1305#issuecomment-554700587
# vpc = ec2.Vpc(self, 'vpc',
#     cidr='10.40.0.0/16',
#     max_azs=2,
#     nat_gateways=2,
#     nat_gateway_provider=ec2.NatProvider.instance(
#         instance_type = ec2.InstanceType('t3a.nano'), # 'instanceType': ec2.InstanceType.of(ec2.InstanceClass.T3A, ec2.InstanceSize.NANO),
#     ),
#     gateway_endpoints= {
#         's3': {'service': ec2.GatewayVpcEndpointAwsService.S3},
#     },
# )

# # Default, using NAT gateway
# vpc = ec2.Vpc(self, 'Vpc', max_azs=2)
