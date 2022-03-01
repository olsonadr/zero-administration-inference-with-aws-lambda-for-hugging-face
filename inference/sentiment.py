"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import json
from transformers import pipeline

nlp = pipeline("sentiment-analysis")

def handler(event, context):
    # Starting point for response formatting
    result = {
        "isBase64Encoded": False,
        "statusCode": 500,
        "headers": { "Content-Type": "application/json" },
        "multiValueHeaders": { },
        "body": ""
    }

    # Get input from body or query string
    if ('text' in event):
        event_text = event['text']
    elif ('body' in event and 'text' in event['body']):
        event_text = event['body']['text']
    elif ('queryStringParameters' in event and 'text' in event['queryStringParameters']):
        event_text = event['queryStringParameters']['text']
    else:
        result['statusCode'] = 500
        result['body'] = "Error! Valid input text not provided!"
        result['headers']['Content-Type'] = "application/json"
        return result

    # Found event_text, use it and return result
    try:
        result['statusCode'] = 200
        result['body'] = nlp(event_text)[0]
        result['headers']['Content-Type'] = "application/json"
    except:
        result['statusCode'] = 500
        result['headers']['Content-Type'] = "application/json"
    return result

    # # Python 3.10 required for this nicer match formatting
    # # Get input from body or query string
    # match event:
    #     # If an expected format
    #     case {'text': event_text} \
    #             | {'body': {'text': event_text}} \
    #             | {'queryStringParameters': {'text': event_text}}:
    #         result['statusCode'] = 200
    #         result['body'] = nlp(event_text)[0]
    #         result['headers']['Content-Type'] = "application/json"
    #         return result
    #     # Else if input not given, return error
    #     case _:
    #         result['statusCode'] = 500
    #         result['body'] = "Error! Valid input text not provided!"
    #         result['headers']['Content-Type'] = "application/json"
    #         return result
