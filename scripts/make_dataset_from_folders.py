

import os
import hashlib
import PIL
import glob
import tqdm
import numpy as np
from imageio import imread

import jsonlines

dataset_path = '/mnt/disks/persist/dataset'

image_directory_paths = glob.glob(f'{dataset_path}/*/')

original_image_paths = []
overlaid_annotation_paths = []
blank_annotation_paths = []
all_generated_captions = []
total = 0


for path in tqdm.tqdm(image_directory_paths):
    samples = [f for f in glob.glob(f'{path}/*.jpg') if '.overlaid' not in f and '.condition' not in f]

    for s in samples:
        annotated_image_path = os.path.splitext(s)[0]+'.overlaid.jpg' 
        
        condition_image_path = os.path.splitext(s)[0]+'.condition.png' 

        with open(os.path.splitext(s)[0]+'.txt', 'r') as f:
            caption = f.readline().strip()

        original_image_paths.append(s)
        overlaid_annotation_paths.append(annotated_image_path)
        blank_annotation_paths.append(condition_image_path)
        all_generated_captions.append(caption)

def gen_examples():
    for i in range(len(original_image_paths)):
        im = imread(blank_annotation_paths[i])
        if not im.sum():
            continue
        else:
            yield {
                "original_image": {"path": original_image_paths[i]},
                "conditioning_image": {"path": blank_annotation_paths[i]},
                "overlaid": {"path": overlaid_annotation_paths[i]},
                "caption": all_generated_captions[i],
            }


with jsonlines.open(f'{dataset_path}/meta.jsonl', 'w') as writer:
    for meta in gen_examples():
        writer.write(meta)
    

"""
final_dataset = Dataset.from_generator(
    gen_examples,
    features=Features(
        original_image=ImageFeature(),
        conditioning_image=ImageFeature(),
        overlaid=ImageFeature(),
        caption=Value("string"),
    ),
    num_proc=6,
)

ds_name = "animalposes-controlnet-dataset"
final_dataset.to_jsonl(f'{dataset_path}/{ds_name}.jsonl')
"""
