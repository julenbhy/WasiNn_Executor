import json
import requests
import time
import base64
import urllib.request
import random


APIHOST='http://172.17.0.1:3233/api/v1'

headers = {'Authorization': 'Basic MjNiYzQ2YjEtNzFmNi00ZWQ1LThjNTQtODE2YWE0ZjhjNTAyOjEyM3pPM3haQ0xyTU42djJCS0sxZFhZRnBYbFBrY2NPRnFtMTJDZEFzTWdSVTRWck5aOWx5R1ZDR3VNREdJd1A='}

def sync_call(action_name: str, params: dict):
    url = APIHOST+'/namespaces/_/actions/'+action_name+'?blocking=true&result=true&workers=1'
    start_time = time.time()
    response = requests.post(url, json=params, headers=headers)
    elapsed_time = time.time() - start_time
    return response.text, elapsed_time

def async_call(action_name: str, params: dict):
    url = APIHOST+'/namespaces/_/actions/'+action_name+'?blocking=false&result=true&workers=1'

    start_time = time.time()
    response = requests.post(url, json=params, headers=headers)
    print('REQUEST:', response.request.__dict__)
    data = json.loads(response.text)
    activation_id = data["activationId"]
    url = APIHOST+'/namespaces/_/activations/'+activation_id

    # Wait until the worker completes the job
    while True:
        result = requests.get(url, headers=headers)
        if result.status_code == 200:
            break
        time.sleep(0.001)
        
    elapsed_time = time.time() - start_time
    result = json.loads(result.text)
    return result['response']['result'], elapsed_time



def main():

    # build the request json
    num_parts = 14

    base_url = "https://huggingface.co/pepecalero/TorchscriptSplitModels/resolve/main/flops-based/gpt2"

    model_links = [f"{base_url}/{num_parts}/{i}.pt" for i in range(num_parts)]

    model_metadata_link = f"{base_url}/{num_parts}/info.json"
    model_metadata = json.loads(requests.get(model_metadata_link).text)
    input_shapes = model_metadata['shapes']

    vocab_url = ["https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",]
    vocab = requests.get(vocab_url[0]).text
    vocab_size = len(json.loads(vocab))

    merges_url = "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt"
    merges = requests.get(merges_url).text

    # Datos para el payload
    text = "Once a vampire fell in love with a pixie so that they"
    max_new_tokens = 100
    top_k = 50

    req_body = {
                'text': text,
                'max_new_tokens': max_new_tokens,
                'top_k': top_k,
                'input_urls': vocab_url,
                'download_method': 'URL',
                'merges': merges,
                'vocab_size': vocab_size,
                'model': model_links, 
                'input_shapes': input_shapes,
                }
    '''
    
    #minio_objects = ['images/2839120406_9b89f48ebf.jpg', 'images/2772193489_c5ff4100fe.jpg',]
    minio_objects = take_random_images('datasets/flicker_urls.txt', 32)
    minio_objects = [f'images/{url.split("/")[-1].strip()}' for url in minio_objects]

    req_body = { 'minio_access_key': 'testuser', 
                 'minio_secret_key': 'testpassworld', 
                 'minio_endpoint': 'http://localhost:9001', 
                 'minio_objects': minio_objects,
                 'model': model_link, 
                 'input_shapes': input_shapes, 
                 'batch_size': minio_objects.__len__(),
                 'class_labels': class_labels,
                 'download_method': 'MinIO'
                }
    '''  




    # make the request
    response ,elapsed_time = sync_call('gpt2_pipeline', req_body)
    print('\nRESPONSE:', response)

    # Print the prediction for each image
    inference = json.loads(response)['result']['inference']
    print('Result:', inference)

    metrics = json.loads(response)['result']['runtime_metrics']
    print('\nMETRICS:', metrics)

    print('\nTIME TAKEN:', elapsed_time)


if __name__ == '__main__':
    main()

