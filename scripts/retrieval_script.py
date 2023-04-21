from img2dataset import download
import shutil
import os
from clip_retrieval.clip_client import ClipClient, Modality
import numpy as np
import csv


def make_query_list(query_file, num_images=1000):
#    query_list = [f"a {animal} {verb}" for animal in ['cat','dog','cow','sheep','horse','panda','deer', 'rhino'] for verb in ['standing', 'walking', 'sitting', 'running']]
    query_list = [f"a photo of a {animal} {verb}" for animal in ['cat','dog'] for verb in ['standing', 'walking']]

    query_file = open('query.csv', 'w')
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=num_images, )
#    query_file =f'{query}.csv'

    writer = csv.writer(query_file)
    writer.writerow(['URL', 'TEXT'])

    for query in query_list:
        results = client.query(text=query)

        for r in results:
            writer.writerow((r['url'], r['caption']))
        
 #   output_dir = os.path.abspath(f"{query}")
    
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)

def download_query_file(query_file, output_dir):
    download(
        processes_count=16,
        thread_count=32,
        url_list=query_file,
        resize_mode='center_crop',
        image_size=512,
        output_folder=output_dir,
        output_format="files",
        input_format="csv",
        url_col="URL",
        caption_col="TEXT",
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        min_image_size=256,
    )
if __name__=="__main__":
    make_query_list('query.csv', 5000)
    download_query_file('query.csv', ' /mnt/disks/persist/basic_dataset/')
