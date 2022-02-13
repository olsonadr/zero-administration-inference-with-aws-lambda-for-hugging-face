"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import json
from transformers import pipeline

summarizer = pipeline("summarization")


def handler(event, context):
    # Starting point for response formatting
    result = {
        "isBase64Encoded": False,
        "statusCode": 500,
        "headers": {"Content-Type": "application/json"},
        "multiValueHeaders": {},
        "body": ""
    }

    # print(event)
    # Get input from body or query string
    if ('text' in event):
        text = event['text']
    elif ('parameters' in event
            and 'text' in event['parameters']):
        text = event['parameters']['text']
    elif ('body' in event
            and 'text' in event['body']):
        text = event['body']['text']
    elif ('queryStringParameters' in event
            and 'text' in event['queryStringParameters']):
        text = event['queryStringParameters']['text']
    # If input not given, return error
    else:
        result['statusCode'] = 500
        result['body'] = "Error! Valid input text not provided!"
        result['headers']['Content-Type'] = "application/json"
        return result
        # return json.dumps(result)

    # Otherwise calculate response and return
    result['statusCode'] = 200
    result['body'] = summarizer(text)[0]
    result['headers']['Content-Type'] = "application/json"
    return result
    # return json.dumps(result)
