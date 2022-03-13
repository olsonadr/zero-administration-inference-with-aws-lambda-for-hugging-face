# AIID NLP Lambdas :hugs:

This project is done in colaboration with the AI Incident Database. Its goal is to support serverless correlation of input texts to similar incidents in the database.

This solution uses the [LongFormer](https://huggingface.co/allenai/longformer-base-4096) model downloaded from [Hugging Face](https://huggingface.co/) as well as the Hugging Face Transformers python library over PyTorch to accomplish ML inference. Hugging Face Transformers is a popular open-source project that provides pre-trained, natural language processing (NLP) models for a wide variety of use cases.

Currently, deployment of this project requires manual configuration of AWS credentials in the [CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html#Prerequisites) (prerequisites section), but future versions will allow for credentials to be configured in GitHub secrets, when GitHub actions is used for CI/CD. The future production version of this scratch repository can be found in the [aiincidentdatabase/nlp-lambdas](https://github.com/aiincidentdatabase/nlp-lambdas) repo.

### Note: This repo is for development scratch and not yet production-ready. What follows is documentation of the current state of this project and plans for future development. The AWS aspects of this solution and aspects of the README layout are adapted from [this Amazon-provided sample project](https://github.com/aws-samples/zero-administration-inference-with-aws-lambda-for-hugging-face).

## Overview

Our solution (will) consist of two major segments:
 - A Python script using a pre-trained LongFormer model and PyTorch to aggregate mean CLS representations for each incident in the AIID database (not currently in a repo, see [Future Development](#Future-Development))
 - An [AWS Cloud Development Kit](https://aws.amazon.com/cdk/) (AWS CDK) script that automatically provisions container image-based Lambda functions that perform ML inference, also using the pre-trained Longformer model. This solution also includes [Amazon Elastic File System](https://aws.amazon.com/efs/) (EFS) storage that is attached to the Lambda functions to cache the pre-trained model and the CLS means of the current DB state that reduces inference latency.

![AWS architecture diagram](serverless-hugging-face-aws-architecture.png)

In this architectural diagram:
 1. Serverless inference (specifically similar-incident resolution) is achieved by using AWS Lambda functions based on Docker container images. 
 2. Each Lambda's docker container contains a saved ```pytorch_model.bin``` file and the necessary configuration files for the a pre-trained LongFormer model, which is loaded from these files by the Lambda on the first execution after deployment, and subsequently cached (in EFS, bullet 5) to accelerate subsequent invocations of the Lambda.
 3. Each Lambda's docker container also contains a pre-processed snapshot of the current state of the AIID database (in the ```incident_cls.pt``` PyTorch file) as a collection of mean CLS representations which are compared against the Longformer's output for the given input text using cosine_similarity to determine similar incidents. Once loaded on first Lambda execution, this representation of the DB state is cached similarly to the model itself (bullet 2).
 4. The container image for the Lambda is stored in an [Amazon Elastic Container Registry](https://aws.amazon.com/ecr/) (ECR) repository within your AWS account.
 5. The pre-trained Longformer model and AIID DB State are cached within Amazon Elastic File System storage in order to improve inference latency.
 6. An HTTP API is generated and hosted using AWS API Gateway to allow the Lambda(s) this project generates to be called by external users and/or future AIID applications. This is (currently) a publically accessible API that can exposes a route for each Lambda (for example, the lambda described in ```similar.py``` is given the route ```/similar```) upon which GET and POST requests can be made, providing input either using URL Query String Parameters (for GET requests) or the request body (for POST requests) as defined in the Lamda's implementation ```.py``` file.

## Prerequisites
The following is required to run this example:
-   [git](https://git-scm.com/)
-   [AWS CDK v2](https://docs.aws.amazon.com/cdk/latest/guide/getting_started.html)
-   [Python](https://www.python.org/) 3.6+
-   [A virtual env](https://docs.python.org/3/library/venv.html#module-venv) (optional)

## Future Development
Going forward we have created a list of features that still need to implemented, created, or integrated. These remaining features will ensure that this project will be easy to access, use, and expand upon.

Remaining Deliverables:
- Consolidate and reconfigure our current code to push to our official Github branch for the AI Incident Database.
  - Ensuring the operation is well documented with good commit logs
  - Integrate CDK initialization with Github Actions
  - Move Large Files to be accessed remotely on an additional Github Branch
  - Replace personal AWS connection with AI Incident Database’s AWS account information
  - Create Github Action to update database information used in CosineSimilarity Calculation
  - Create Github Action to trigger our Test Suite
- Create a fully-fledged and all-encompassing Test Suite
  - Use an official Python testing suite to create tests that test the entire process and each part individually to ensure all operations are completed as expected

## Deploying the example application
1.  Clone the project to your development environment:
    ```bash
    git clone <https://github.com/olsonadr/zero-administration-inference-with-aws-lambda-for-hugging-face.git>
    ```

2.  Download the ```pytorch_model.bin``` and ```incident_cls.pt``` files from the [current release](https://github.com/olsonadr/zero-administration-inference-with-aws-lambda-for-hugging-face/releases) of this repo and place them in the ```inference/model``` directory (this workflow will be replaced with LFS later, but public forks do not allow this at this time).

3.  Set all required environment variables directly or in a ```.env``` file (future versions will use Github Actions for CI/CD and will use Github secrets for this):
    - ```AWS_ACCOUNT_ID```: the Account ID for the AWS account that will host the AWS application stack
    - ```AWS_REGION```: the AWS server region to deploy the AWS application stack on (i.e. ```us-west-2```)

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Bootstrap the CDK. This command provisions the initial resources
    needed by the CDK to perform deployments:
    ```bash
    cdk bootstrap
    ```

5.  This command deploys the CDK application to its environment. During
    the deployment, the toolkit outputs progress indications:
    ```bash
    cdk deploy
    ```

## Understanding the code structure
The code is organized using the following structure (only relevant files shown, \* explained below):
```bash
├── inference
│   ├── model
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── tokenizer.json
│   │   ├── *pytorch_model.bin
│   │   └── *incident_cls.pt
│   ├── Dockerfile
│   └── similar.py
├── app.py
└── ...
```

- The ```app.py``` script is explained in the [CDK Script](#CDK-Script) section. It describes the full AWS stack of this solution and is run using the [AWS CDK v2](https://docs.aws.amazon.com/cdk/latest/) command-line to deploy the stack to AWS servers.
- The ```inference``` directory contains the files that constitute each AWS Lambda and their Docker configuration. It specifically contains:
    -   The ```Dockerfile``` used to build a custom image to be able to run PyTorch Hugging Face inference using Lambda functions and that adds the current LongFormer Model and CLS Means in the ```inference/model``` directory into the container for each lambda
    -   The Python scripts that perform the actual ML inference (```similar.py```)
    -   The ```model``` directory, which contains:
        -   The ```config.json```, ```merges.txt```, and ```tokenizer.json``` HuggingFace boilerplate of the currently used version of the [Longformer Model HuggingFace Repo](https://huggingface.co/allenai/longformer-base-4096/tree/main)
        -   The ```pytorch_model.bin``` model file of the currently used version of the [Longformer Model HuggingFace Repo](https://huggingface.co/allenai/longformer-base-4096/tree/main). This is starred because this file is contained in the [current release](https://github.com/olsonadr/zero-administration-inference-with-aws-lambda-for-hugging-face/releases) of this repo and must be manually downloaded and placed here. This file is **required** for correct execution
        -   The ```incident_cls.pt``` current AIID DB State CLS Means used for cosine similarity comparisons with input text. This is starred because this file is contained in the [current release](https://github.com/olsonadr/zero-administration-inference-with-aws-lambda-for-hugging-face/releases) of this repo and must be manually downloaded and placed here. This file is **required** for correct execution

The ```similar.py``` script shows how to use a local model file contained in the Docker conatainer (note the ```local_files_only``` option usage) and handle event input and output w/ the AWS Lambda<->API Proxy-Integration:

```python
# Paraphrased for simplicity (esp. no error handling)
import os
import json
import torch
from transformers import LongformerTokenizer, LongformerModel
from unidecode import unidecode

# Load NLP models and state from files in container and store
#    global vars in EFS cache for faster subsequent execution
model = LongformerModel.from_pretrained('/function/model',
                                        local_files_only=True)
tokenizer = LongformerTokenizer.from_pretrained('/function/model',
                                                local_files_only=True)
tensors = torch.load('/function/model/incident_cls.pt')

def process(text, tensors):
    inp = tokenizer(text, padding="longest",
                    truncation="longest_first",
                    return_tensors="pt")
    out = model(**inp)
    similarities = [
        torch.nn.functional.cosine_similarity(out.last_hidden_state[0][0],
                                              tensors[i].mean(0), 
                                              dim=-1
                                             ).item()
        if tensors[i] != None else torch.zeros(1)
        for i in range(len(tensors))
    ]
    return [j for j in sorted(zip(similarities, range(1, len(tensors) + 1)), reverse=True)]

def handler(event, context):
    result = {
        "isBase64Encoded": False,
        "statusCode": 500,
        "headers": { "Content-Type": "application/json" },
        "multiValueHeaders": { },
        "body": ""
    }
    
    event_text = json.loads(event['body'])['text'] # for post requests
    event_text = unidecode(event_text) # handle unicode input
    
    result['body']['msg'] = str(process(event_text, best_of))
    result['statusCode'] = 200
    
    return json.dumps(result)
```

For each Python script in the inference directory, the CDK generates a
Lambda function backed by a container image and a Python inference
script.

## Longformer Model
When a call is made to the API, the input text is tokenized and processed by the Longformer Model. The output is a tensor representation. The Special Classification (CLS) token is the first vector in this processed representation, and contains some latent information about the input text.

The CLS vector is then compared, using cosine similarity, to the current stored state for each incident in the Incident Database. The incident state is the mean of the preprocessed CLS tokens of all reports for that incident. Updating this state will later be done automatically via GitHub Actions, either when reports are added to the Incident Database, or on a regular basis, e.g. once per day.

The output of the comparison is a list of confidence scores, describing the input text's similarity to each incident. The highest N of these are the most similar incidents to the input text, and are sent as the response to the API call.

## API Documentation
The AWS HTTP API this CDK application creates uses a the [Lambda Proxy-Integration](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html) standard for requests to and from the Lambdas. This necessitates specific [input](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format) and [output](api-gateway-simple-proxy-for-lambda-output-format) formats between the Lambda and the API. This format is used in the ```similar.py``` Lambda function implementation.

This is different than the request and response format specified for API Gateway <-> User communications. For these transactions, the current version of this API and Lambda Stack expects the following request and respone formats:

### ```similar.py``` API req/res formats (most similar incidents in AIID DB to input text)

#### ```similar.py``` parameters for **all** request types (currently **GET** & **POST**)
 - API endpoint: ```https:[API_URL]/similar```
 - Input variables:
     - ```text```: *required*, the input text to process
     - ```num```: *optional (default 3)*, the number of most-similar incidents to return (or -1 for all incidents in DB, ranked)
 - Relevant output variables (in HTTP response):
     - ```statusCode```: the [HTTP status code](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) (i.e. 200 success, 500 error, etc.)
     - ```warnings```: list of any warning messages returned from the Lambda (incorrect but recoverable request formatting, etc.)
     - ```msg```: the requested output from the Lambda (i.e. list of tuples with similarity score and ID of that incident) or a Lambda-specified error message
 - HTTP response format:
     - Response format with output variable names as placeholders  (placeholders surrounded by **)
        ```json
        {
            "isBase64Encoded": false,
            "statusCode": *statusCode*,
            "headers": {
                "Content-Type": "application/json"
            },
            "multiValueHeaders": {},
            "body": {
                "warnings": *warnings*,
                "msg": *msg*
            }
        }
        ```
    - Response example with example values for outputs 
        ```json
        {
            "isBase64Encoded": false,
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "multiValueHeaders": {},
            "body": {
                "warnings": ["Provided value for \"num\" invalid, using default of 3."],
                "msg": "[(0.9975811839103699, 1), (0.996882975101471, 55), (0.9966274499893188, 39)]"
            }
        }
        ```
        
#### ```similar.py``` **GET** request specifics
 - Request format (uses URL query string parameters):
     - Request example with input variable names as placeholders  (placeholders surrounded by \*\*)
    ```https:[API_URL]/similar?num=*num*&text="*text*"```
     - Request example with example values (request all incidents for text "wow this is the body of a news article"):
    ```https:[API_URL]/similar?num=-1&text="Wow, this is the body of a news article!"```
     - Request example with example values (default ```num``` of 3 most similar incidents for text "wow this is the body of a news article"):
    ```https:[API_URL]/similar?num=-1&text="Wow, this is the body of a news article!"```
    

#### ```similar.py``` **POST** request specifics
 - Request body content format with input variable names as placeholders  (placeholders surrounded by **)
    ```json
    {
      "text": "*text*",
      "num": *num*
    }
    ```
 - Request example with example values (request all incidents for text "wow this is the body of a news article"):
    ```json
    {
      "text": "wow this is the body of a news article",
      "num": -1
    }
    ```
- Request example with example values (default ```num``` of 3 most similar incidents for text "wow this is the body of a news article"):
    ```json
    {
      "text": "wow this is the body of a news article"
    }
    ```

## CDK script
The CDK script is named ```app.py``` in the solution's repository. This script describes the AWS stack for the Lambda solution using the CDK V2 python library, to enable the CDK CLI application to facillitate deployment to AWS servers. The important segments of this script are described below:

The beginning of the script (global scope) imports the necessary libraries, allows the use of ```.env``` environment variables, and establishes the AWS account and region to be used for the stack:
```python
import os;
from dotenv import load_dotenv
from pathlib import Path
from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigatewayv2_alpha as api_gw2_a, # apigatewayv2 is in alpha stage
    aws_efs as efs,
    aws_ec2 as ec2
)
from aws_cdk import (App, Stack, Duration, RemovalPolicy,
                     Environment, CfnCondition, Fn, CfnOutput)
from aws_cdk.aws_apigatewayv2_integrations_alpha import HttpLambdaIntegration # apigatewayv2 is in alpha stage
from constructs import Construct

load_dotenv()
aws_env = Environment(account=os.environ['AWS_ACCOUNT_ID'],
                      region=os.environ['AWS_REGION'])
```

The remainder of the script is contaiend within the ```__init__``` method of the ```ServerlessHuggingFaceStack``` AWS stack:
```python
# Stack for this cdk program
class ServerlessHuggingFaceStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        ...
```

The beginning of this stack description creates a virtual private cloud (VPC) that is necessary for the housing the EFS for caching. The current version of this solution intentionally uses an [Internet Gateway]() rather than a [NAT Gateway](). This means that there is no internet access within the lambda execution like there would be with a NAT Gateway, but also reduces cost of hosting from $2 a day (at time of writing) to $0.01 a day. The following code examples shows how to do each of these:
- **Recommended** VPC with no NAT Gateway (using Fargate Internet Gateway with public subnet):
    ```python
    # Virtual private cloud with no NAT Gateway (Fargate Internet Gateway with public subnet):
    vpc = ec2.Vpc(self, 'FargateVPC',
                  cidr='10.0.0.0/16',
                  max_azs=1,
                  nat_gateways=0,
                  subnet_configuration=[
                      {'cidr_mask': 24,
                       'name': 'Public',
                       'subnetType': ec2.SubnetType.PUBLIC }
                  ])
    ``` 
- **OR** VPC w/ the default NAT Gateway (and private subnet, **extra costs**):
    ```python
    # **OR** Virtual private cloud with the default NAT Gateway (and private subnet):
    vpc = ec2.Vpc(self, 'Vpc', max_azs=2)
    ```
    
Next, the CDK script creates the EFS file system and an access point in EFS for the cached model:
```python
# Create a file system in EFS to store cache models
fs = efs.FileSystem(self, 'FileSystem',
                    vpc=vpc,
                    removal_policy=RemovalPolicy.DESTROY)
access_point = fs.add_access_point('MLAccessPoint',
                                   create_acl=efs.Acl(
                                       owner_gid='1001', owner_uid='1001', permissions='750'
                                   ),
                                   path="/export/models",
                                   posix_user=efs.PosixUser(gid="1001", uid="1001"))
```

Next, the CDK script creates the Http API Gateway for this stack (the lambda integrations and routes on this API are added later):
```python
# Create Http Gateway API for this stack
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
                              'allow_origins': ['http://localhost:3000'],
                            }
                           )

# Create a CDK output for the api url
# this outputs the apiUrl to a json file when deploying the stack, so it can be used by other programs as needed
CfnOutput(self, 'apiUrl', value=httpApi.url)
```

It then iterates through the Python files in the inference directory:
```python
docker_folder = os.path.dirname(
    os.path.realpath(__file__)) + "/inference"
pathlist = Path(docker_folder).rglob('*.py')
for path in pathlist
    ...
```

And for each lambda ```.py``` file found, it creates a Lambda function that serves the inference requests and adds this lambda as a route on the API with a Lambda Proxy Integration:
```python
# Parse lambda handler filename name
base = os.path.basename(path)
filename = os.path.splitext(base)[0]

# Lambda Function from docker image
function = lambda_.DockerImageFunction(
    self, filename,
    allow_public_subnet=True, # important for Internet Gateway
    code=lambda_.DockerImageCode.from_image_asset(
        docker_folder, cmd=[filename+".handler"]),
    memory_size=8096,
    timeout=Duration.seconds(600),
    vpc=vpc,
    filesystem=lambda_.FileSystem.from_efs_access_point(
        access_point, '/mnt/hf_models_cache'), # Establish cache
    environment={ # example environment variables that you could use in the lambda scripts after being set here
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

# Add route lambda integration to a route on the api, using the fn 
#     name and allowing for both GET and POST requests to this lambda
httpApi.add_routes(
    path=f"/{filename}",
    methods=[api_gw2_a.HttpMethod.GET, api_gw2_a.HttpMethod.POST],
    integration=lambda_integration
)
```

After the Stack is defined, back at the global scope, the CDK script concludes by initializing the AWS stack:
```python
# Initialize the aws stack
app = App()
ServerlessHuggingFaceStack(app,
                           "ServerlessHuggingFaceStack",
                           env=aws_env)
app.synth()
```

## Adding additional Lambdas to the Stack
Optionally, you can add more models by adding Python scripts in the ```inference``` directory. For example, the sample script ```inference/sentiment.py``` shows how you could download and use a model from HuggingFace for sentiment analysis (would work **if and only if** you replace the internet gateway currently used with a NAT gateway -- [instructions in CDK Script section](#CDK-Script)) and without using the AWS Proxy-Integration request format:

```python
# Paraphrased for simplicity (esp. no error handling)
import json
from transformers import pipeline

# Download model from HuggingFace and store global vars in EFS cache
nlp = pipeline("sentiment-analysis")

def handler(event, context):
    result = {
        "isBase64Encoded": False,
        "statusCode": 500,
        "headers": { "Content-Type": "application/json" },
        "multiValueHeaders": { },
        "body": ""
    }
    
    result['body'] = nlp(event['body']['text'])[0]
    result['statusCode'] = 200
    
    return result
```
Then run:
```bash
$ cdk synth
$ cdk deploy
```
This creates a new lambda function to perform sentiment analysis *(although you must copy the Proxy request and response structures to use this lambda with the Http API Gateway)*.

## Cleaning up
After you are finished experimenting with this project, run ```cdk destroy``` to remove all of the associated infrastructure locally and on the AWS servers. If you do not do this, and especially if you are using the NAT Gateway, you will accrue AWS charges while the Stack is hosted.

## License
This library is licensed under the MIT No Attribution License. See the [LICENSE](LICENSE) file.

Disclaimer: Deploying the demo applications contained in this repository will potentially cause your AWS Account to be billed for services.

## Links
- [:hugs:](https://huggingface.co)
- [AWS Cloud Development Kit](https://aws.amazon.com/cdk/)
- [Amazon Elastic Container Registry](https://aws.amazon.com/ecr/)
- [AWS Lambda](https://aws.amazon.com/lambda/)
- [Amazon Elastic File System](https://aws.amazon.com/efs/)
