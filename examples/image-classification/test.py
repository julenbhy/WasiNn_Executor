import json
import requests
import time
import base64
import urllib.request
import random
import os
import subprocess


APIHOST='http://172.17.0.1:3233/api/v1'

headers = {'Authorization': 'Basic MjNiYzQ2YjEtNzFmNi00ZWQ1LThjNTQtODE2YWE0ZjhjNTAyOjEyM3pPM3haQ0xyTU42djJCS0sxZFhZRnBYbFBrY2NPRnFtMTJDZEFzTWdSVTRWck5aOWx5R1ZDR3VNREdJd1A='}

def sync_call(action_name: str, params: dict):
    url = APIHOST+'/namespaces/_/actions/'+action_name+'?blocking=true&result=true&workers=1'
    start_time = time.time()
    response = requests.post(url, json=params, headers=headers)
    elapsed_time = time.time() - start_time
    return response.text, elapsed_time


def update_action(action_name: str, zip_path: str, urls: list[str], parameters: list[str]):
    """
    Updates an OpenWhisk action with the given filename, model URLs and parameters.

    Args:
        filename (str): The name of the action to update.
        urls (list[str]): A list of model URLs.
        parameters (list[str]): A list of parameters.
    """
    # Prepare paths and strings
    url_str = json.dumps(urls)  # '["https://...", ...]'
    param_str = json.dumps(parameters)  # '["param1", ...]'

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Action zip not found: {zip_path}")

    cmd = [
        "wsk", "action", "update",
        "--kind", "wasm:0.1",
        action_name,
        zip_path,
        "-a", "model_urls", url_str,
        "-a", "parameters", param_str,
        #"-p", "parameters", param_str
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Action '{action_name}' updated.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to update action '{action_name}': {e}")



def take_random_images(path, num_images):
    with open(path) as f:
        lines = f.readlines()
    return random.sample(lines, num_images)

def main():

    action_name = 'pytorch_image_classification'

    # build the request json
    num_parts = 3

    base_url = "https://huggingface.co/pepecalero/TorchscriptSplitModels/resolve/main/flops-based/squeezenet1_1"
    #base_url = "https://huggingface.co/pepecalero/TorchscriptSplitModels/resolve/main/flops-based/resnet50"

    model_urls = [f"{base_url}/{num_parts}/{i}.pt" for i in range(num_parts)]



    # Build the url to the model metadata
    model_metadata_url = f"{base_url}/{num_parts}/info.json"
    model_metadata = json.loads(requests.get(model_metadata_url).text)
    input_shapes = model_metadata['shapes']

    classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classes_url_response = urllib.request.urlopen(classes_url)
    class_labels = [line.strip() for line in classes_url_response.read().decode("utf-8").split("\n") if line]

    parameters = { 'input_shapes': input_shapes,
                  #'batch_size': image_urls.__len__(),
                   'batch_size': 4,
                   'class_labels': class_labels,
                }


    # Update the action with the new model URLs
    print(f"Updating action '{action_name}' with {len(model_urls)} model URLs")
    update_action(action_name, f"actions/compiled/{action_name}.zip", model_urls, parameters)


    ### TEST FROM FLICKER DATASET ###
    # Get a random set of images from the flicker dataset
    #num_images = 3
    #image_urls = take_random_images('datasets/flicker_urls.txt', num_images)

    # Hardcoded image URLs for testing
    image_urls = [  "http://static.flickr.com/3020/2772193489_c5ff4100fe.jpg", # ballplayer, racket, scoreboard
                    "http://farm4.static.flickr.com/3227/2790859586_e95672cabf.jpg", # ballplayer, racket, basketball
                    "http://farm1.static.flickr.com/145/334514409_f024092d2f.jpg?v=0", # volleyball, horizontal bar, basketball
                    "http://farm3.static.flickr.com/2332/2442000320_bd159cb510.jpg?v=0", # volleyball, racket, basketball
                    "http://farm4.static.flickr.com/3203/2853289237_3724173dc7.jpg", # cock, hen, proboscis monkey
                    "http://farm1.static.flickr.com/9/76367251_d41e29386c.jpg", # spoonbill, black stork, pelican
                    "http://farm1.static.flickr.com/162/333803097_b56538e5de.jpg", # tench, barracouta, gar
                    "http://farm4.static.flickr.com/3196/3050927387_6374cedf8c.jpg", # goldfish, coral reef, jellyfish
                    #"http://farm3.static.flickr.com/2030/1505549474_5670617cf7.jpg", # great white shark, grey whale, leatherback turtle
    ]
    #num_images = 32
    #image_urls = take_random_images('datasets/flicker_urls.txt', num_images)

    req_body = { 'input_urls': image_urls,
                 'download_method': 'URL',
                 'batch_size': 4
               }


    # Call the pipeline action with the tensor optimization
    print(f"\nCalling action '{action_name}' with {len(image_urls)} images")
    response ,elapsed_time = sync_call(action_name, req_body)
    #print('\nRESPONSE:', response)
    inference = json.loads(response)['result']['inference']
    for key in inference:
        batch = inference[key]
        print('\n', key, ':')
        for key2 in batch:
            print('\n\t', key2, ':', batch[key2])
    


    # Sleep 2 seconds and perform a second call to the action
    #'''
    time.sleep(2)
    print(f"\nCalling action '{action_name}' with {len(image_urls)} images second time")
    response ,elapsed_time = sync_call(action_name, req_body)
    #print('\nRESPONSE:', response)
    inference = json.loads(response)['result']['inference']
    for key in inference:
        batch = inference[key]
        print('\n', key, ':')
        for key2 in batch:
            print('\n\t', key2, ':', batch[key2])
    
    #'''




    metrics = json.loads(response)['result']['metrics']
    print('\nMETRICS:', metrics)

    print('\nTIME TAKEN:', elapsed_time)


if __name__ == '__main__':
    main()

