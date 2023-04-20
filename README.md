# animalpose


## Getting started
1) Read instructions here:    
https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint#data-and-pre-processing   
2) Getting a feel of the openpifpaf animal keypoint detection model.
3) Start to build the library of triplets (image, keypoints images, caption).    
4) Try to run things on TPU, read the docs if any issues.     

## Methods
### Controlnet
200k pose- image- caption pairs were used for the original human pose controlNet.
They also set a threshold of 30%of keypoints being detected to use the image.
See:
https://arxiv.org/abs/2302.05543
We need to aim to something of this order of magnitude    

## Datasets
### General datasets
#### Laion 400m
Can be accessed via
https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md    

You can actually filter by caption.

Good results from delirious using that. Much more consistency in detection of keypoints. Really promising.

#### Coco 2014 dataset
https://cocodataset.org/#home
You can build a filter for animals
`def filter_fn(x):`
    `return tf.reduce_any(tf.equal(x['objects']['label'], 19))`
`cows = data.filter(filter_fn)`
14 is bird, 15 is cat, 16 is dog, 17 is horse, 18 is sheep, 19 is cow, 20 is elephant, 21 is bear, 22 is zebra, 23 is giraffe    
This is a much smaller dataset than Laion400m


### Animal specific datasets
#### Animal pose dataset
https://github.com/kfahn22/animalpose/tree/animal_pose/Animal%20Pose%20Dataset
See summary from kfahn22 in branch:
https://github.com/kfahn22/animalpose/tree/animal_pose/Animal%20Pose%20Dataset

### Species specific datasets
### Dog datasets
1) Tsinghua
https://cg.cs.tsinghua.edu.cn/ThuDogs/
70428 images of dogs

2) Standford dog dataset
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset   
20000 images of dogs

## Results and discussion
### First result from Jafoz are now on hugginface
https://huggingface.co/JFoz/dog-pose    
This was trained on 6k image-pose-caption pairs at:
https://huggingface.co/datasets/JFoz/dog-poses-controlnet-dataset
The images are from the standford dog dataset
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
Possible overfitting. Regardless 6,000 images is much smaller than what was used for the human pose dataset. This is a good start.

## Next steps

1) Use a much bigger datasets to obtain a image-keypoint-caption pair dataset an order of magnitude larger than the previous one
2) Use this to train a new animal controlnet
3) Investigate the possibility of finetuning the existing openpifpaf human controlNet for animals

