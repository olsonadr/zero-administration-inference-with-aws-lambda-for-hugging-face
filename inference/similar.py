"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

# Imports
import os
import json
import torch
from transformers import LongformerTokenizer, LongformerModel
from unidecode import unidecode

def list_files(startpath):
    res = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        res = res + ('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            res = res + ('{}{}'.format(subindent, f))
    return res


# # Get model and tokenizer from EFS or download it to EFS
# model_path = os.path.join(
#     os.environ["TRANSFORMERS_CACHE"], os.environ['MODEL_DIR'], os.environ["MODEL_FILENAME"])
# model_dir = os.path.join(
#     os.environ["TRANSFORMERS_CACHE"], os.environ['MODEL_DIR'])
# hf_uri = os.environ["HF_MODEL_URI"]

# # If it exists in the EFS, load from EFS
# if os.path.isfile(model_path):
#     tokenizer = LongformerTokenizer.from_pretrained(hf_uri)
#     model = LongformerModel.from_pretrained(hf_uri)
#     tokenizer.save_pretrained(model_dir)
#     model.save_pretrained(model_dir)
# # Else, not saved into EFS yet, get from hf and save
# else:
#     tokenizer = LongformerTokenizer.from_pretrained(
#         model_dir, local_files_only=True)
#     model = LongformerModel.from_pretrained(model_dir, local_files_only=True)

# # Constants
# incidents_path = os.path.join(
#     os.environ["TRANSFORMERS_CACHE"], os.environ["INCIDENTS_FILENAME"])
# csv_path = os.path.join(
#     os.environ["TRANSFORMERS_CACHE"], os.environ["CSV_FILENAME"])
model = LongformerModel.from_pretrained(
    '/function/model', local_files_only=True)
tokenizer = LongformerTokenizer.from_pretrained(
    '/function/model', local_files_only=True)
csv_path = '/function/model/incidents.csv'
incidents_path = '/function/model/incident_cls.pt'
best_of = 3

# Load in a list of articles from a CSV
tensors = torch.load(incidents_path)
# data = read_csv(csv_path)

# Test effacacy of preprocessing and meaning whole output rather than
# stripping CLS token. (Cursory: much more effective)


def test(text):
    inp = tokenizer(text,
                    padding="longest",
                    truncation="longest_first",
                    return_tensors="pt")
    out = model(**inp)
    sims = [
        torch.nn.functional.cosine_similarity(out.last_hidden_state[0][0],
                                              tensors[i].mean(0),
                                              dim=-1).item()
        if tensors[i] != None else torch.zeros(1)
        for i in range(len(tensors))
    ]
    return sims


def inputted(whole_text):
    sims = [j for j in sorted(
        zip(test(whole_text), range(1, len(tensors) + 1)), reverse=True)]
    best = sims[:best_of]
    return best
    # return sims

# What to do to correctly formatted input event_text
def process(event_text):
    # return tokenizer(event_text)
    return str(inputted(event_text))

# Define lambda handler
def handler(event, context):
    # Starting point for response formatting
    result = {
        "isBase64Encoded": False,
        "statusCode": 500,
        "headers": {"Content-Type": "application/json"},
        "multiValueHeaders": {},
        "body": ""
    }

    # Get input from body or query string
    if ('text' in event):
        event_text = event['text']
    elif ('body' in event and event['body'] != '' and 'text' in json.loads(event['body'])):
        event_text = json.loads(event['body'])['text']
    elif ('queryStringParameters' in event and 'text' in event['queryStringParameters']):
        event_text = event['queryStringParameters']['text']
    else:
        result['statusCode'] = 500
        result['body'] = {'msg': 'Error! Valid input text not provided!'}
        result['headers']['Content-Type'] = "application/json"
        return json.dumps(result)
        # return result

    # Handle unicode in event_text
    event_text = unidecode(event_text[:6000])

    # Found event_text, use it and return result
    try:
        result['statusCode'] = 200
        result['body'] = {'msg': process(event_text)}
        result['headers']['Content-Type'] = "application/json"
    except:
        result['statusCode'] = 500
        result['headers']['Content-Type'] = "application/json"
    return json.dumps(result)
    # return result

    # # Python 3.10 required for this nicer match formatting (not updated for proxy integration)
    # # Get input from body or query string
    # match event:
    #     # If an expected format
    #     case {'text': event_text} \
    #             | {'body': {'text': event_text}} \
    #             | {'queryStringParameters': {'text': event_text}}:
    #         result['statusCode'] = 200
    #         # result['body'] = nlp(event_text)[0]
    #         result['body'] = inputted(event_text)
    #         result['headers']['Content-Type'] = "application/json"
    #         return result
    #     # Else if input not given, return error
    #     case _:
    #         result['statusCode'] = 500
    #         result['body'] = "Error! Valid input text not provided!"
    #         result['headers']['Content-Type'] = "application/json"
    #         return result
