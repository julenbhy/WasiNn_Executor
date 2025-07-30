import json
import requests
import time
import base64
import urllib.request
import random
import os
import subprocess
from IPython.display import display
from itertools import product




action_name = 'pytorch_image_classification'
input_urls_file = 'datasets/flicker_urls.txt' 
num_images_list = [256, 512, 1024]
batch_sizes = [32, 64, 128]
num_iters = 2



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


import pandas as pd

def benchmark(req_body, num_iters, num_images, input_urls_file):
    metrics_list = []  # Lista para almacenar métricas de cada iteración

    for i in range(num_iters):
        print(f"\nITERATION {i + 1}")
        try:
            input_urls = take_random_images(input_urls_file, num_images)
            req_body["input_urls"] = input_urls

            response ,elapsed_time = sync_call(action_name, req_body)
            #print(f"\nIteration {i + 1} RESPONSE:", response)
            
            inference = json.loads(response)['result']['inference']
            #print(f"\nIteration {i + 1} 'inference':", inference)
            metrics = json.loads(response)['result']['metrics']
            print(f"\nIteration {i + 1} 'metrics':", metrics)

            # Convert lists to separate columns (flatten)
            flattened_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, list):
                    for idx, v in enumerate(value):
                        flattened_metrics[f"part_{idx}_{key}"] = v
                else:
                    flattened_metrics[key] = value

            metrics_list.append(flattened_metrics)
            

        except Exception as e:
            print("\nERROR on iteration", i + 1, ":", e)

        # Add a delay to avoid overwhelming the server
        time.sleep(15)

    df_metrics = pd.DataFrame(metrics_list)

    return df_metrics



def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config



def main():
    results_dir = 'tensor_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    config = load_config("config_small.json")

    # Load class labels
    classes_url = config['imagenet_classes_url']
    classes_url_response = urllib.request.urlopen(classes_url)
    class_labels = [line.strip() for line in classes_url_response.read().decode("utf-8").split("\n") if line]

    # Split method, model, part
    items = [
        (split_method, model, part)
        for split_method, models in config.items() if split_method != "imagenet_classes_url"
        for model, parts in models.items()
        for part in parts
    ]

    # Iterate over all split methods, models and parts
    for split_method, model, part in items:
        print(f'\nRUNING: {split_method} {model} {part}')

        model_urls = config[split_method][model][part]['model_urls']

        model_metadata_url = config[split_method][model][part]['model_metadata_url']
        model_metadata = json.loads(requests.get(model_metadata_url).text)
        input_shapes = model_metadata['shapes']

        parameters_template = {'input_shapes': input_shapes, 
                             'class_labels': class_labels,
                            }

        #print('\nWARMING UP...')
        #req_body = parameters_template.copy()
        #req_body['batch_size'] = 16
        #df_warmup = benchmark(req_body, 3, 32, input_urls_file)

        # Create directory structure
        part_dir = os.path.join(results_dir, split_method, model, part)
        os.makedirs(part_dir, exist_ok=True)

        for num_images in num_images_list:
            for batch_size in batch_sizes:

                # Skip if batch size is greater than number of images
                if batch_size > num_images:
                    print(f'Skipping batch size {batch_size} for num_images {num_images} (batch size > num images)')
                    continue


                # Update the action with the new model URLs
                print(f"UPDATING ACTION '{action_name}'")
                parameters = parameters_template.copy()
                parameters['batch_size'] = batch_size
                update_action(action_name, f"../../actions/compiled/{action_name}.zip", model_urls, parameters)


                print(f'\nBENCHMARKING NUM_IMAGES {num_images} BATCH_SIZE {batch_size}...')

                req_body = {'download_method': 'URL',
                            'batch_size': batch_size}
                
                print('\nWARMING UP...')
                df_warmup = benchmark(req_body, 1, batch_size, input_urls_file)

                print('\nSTARTING BENCHMARK...')
                df = benchmark(req_body, num_iters, num_images, input_urls_file)

                # Check if df is empty or not
                if df.empty:
                    print(f"Warning: DataFrame is empty for num_images {num_images} batch_size {batch_size}")
                    continue

                numeric_cols = df.select_dtypes(include=['number']).columns
                df.loc['avg', numeric_cols] = df[numeric_cols].mean()
                filename = os.path.join(part_dir, f'num_images_{num_images}_batch_{batch_size}.csv')
                df.to_csv(filename)


if __name__ == '__main__':
    main()

