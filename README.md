# Image Corpus Client

The image corpus client is a python package providing the primitives to interract with Max-ICS Corpus API
# Usage
```bash
usage: corpus_download [-h] --user USER --password PASSWORD [--verbose]
                       [--test-size TEST_SIZE] [--width WIDTH]
                       [--height HEIGHT] [--drop-smaller] [--staging]
                       corpus_id

Download a corpus to disk

positional arguments:
  corpus_id             the corpus id

optional arguments:
  -h, --help            show this help message and exit
  --user USER, -u USER  user for auth
  --password PASSWORD, -p PASSWORD
                        password for auth
  --verbose, -v         Debug mode
  --test-size TEST_SIZE
                        fraction of data in the test set. Default to 0.15
  --width WIDTH         Minimum width of image. default to 500
  --height HEIGHT       Minimum height of image. default to 500
  --drop-smaller        drop image smaller than size
  --tile_x TILE_X       Number of tile on the x axis. default to 1
  --tile_y TILE_Y       Number of tile on the y axis. default to 1
  --staging             use the staging env
```

# Classification corpus
In case the corpus is of type *'CLASSIFICATION'*, it will download it according to the keras guideline. 
It will have the following format : 
```text
corpus_name/
    dog/
        image001.jpg
        image002.jpg
    cat/
        image003.jpg
        image004.jpg
```
It is directly compatible with keras `flow_from_diretory()` method. If used with this Keras method, the label index in the prediction will be the sorted names of the folders. 
Like `cat: 0, dog: 1` 
**Note that an image cannot have multiple class in this case.**

# Semantic segmentation
In this case, corpus are downloaded in an unique .h5 (hdf5) file. 
You can use [h5py](http://www.h5py.org/) to parse them.
The downloaded h5 file have the following structure :
```text
root
    train: Group containing the train datasets
        data: Dataset containing the ground truth image. Shape: [dataset_length, image_height, image_width, image_channel]
        label: Dataset containing the ground truth labels. Shape: [dataset_length, image_height, image_width, number_of_classes]
    eval: Group containing the eval datasets
        data: Dataset containing the ground truth image. Shape: [dataset_length, image_height, image_width, image_channel]
        label: Dataset containing the ground truth labels. Shape: [dataset_length, image_height, image_width, number_of_classes]
    label-map : Attribute (of the root hdf5 file) containing the label map
```
the label map will be like `{"dog":0, "cat":1}` with the same ordering than the corpus.

# Note
## Tiling
you can create tile of the input image in the dataset. For instance, if you set `--width 1024 --height 1024 --tile_x 2 --tile_y 2` you will end with 4 images in the h5 file per image in the
dataset. In this case, each of the image in the dataset will of size *512x512*.  
Behaviour of the label ill depend on the corpus type:
* Classification: each of the tiles will have the same label.
* Semantic seg: the label matrix will also be tiled

## h5py example
```python
import json
import h5py
from matplotlib import pyplot as plt

f = h5py.File('/path/to/corpus.h5', 'r') # to be sure to be in read-only mode
label_map = json.loads(f.attrs['label-map'])
print(label_map)

for i in range(10):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(f['eval/data'][i])
    plt.subplot(122)
    plt.imshow(f['eval/label'][i,:,:, label_map['dog']], cmap='gray')
    plt.show() 
f.close()
```