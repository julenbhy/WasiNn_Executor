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



def take_random_images(path, num_images):
    with open(path) as f:
        lines = f.readlines()
    return random.sample(lines, num_images)

def main():
    # build the request json
    num_parts = 2

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
                    "http://farm3.static.flickr.com/2030/1505549474_5670617cf7.jpg", # great white shark, grey whale, leatherback turtle
    ]
    num_images = 32
    image_urls = take_random_images('datasets/flicker_urls.txt', num_images)

    req_body = { 'input_urls': image_urls,
                 'download_method': 'URL',
                 'model': model_urls, 
                 'input_shapes': input_shapes, 
                 #'batch_size': image_urls.__len__(),
                 'batch_size': 7,
                 'class_labels': class_labels,
                }
    

    # Call the pipeline action with the tensor optimization
    response ,elapsed_time = sync_call('pytorch_image_classification', req_body)
    print('\nRESPONSE:', response)
    inference = json.loads(response)['result']['inference']
    for key in inference:
        batch = inference[key]
        print('\n', key, ':')
        for key2 in batch:
            print('\n\t', key2, ':', batch[key2])
    


    metrics = json.loads(response)['result']['metrics']
    print('\nMETRICS:', metrics)

    print('\nTIME TAKEN:', elapsed_time)


if __name__ == '__main__':
    main()

