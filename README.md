# K9 License Plate Recognition - ML part

This repository is a part of an project prepared for *Team Programming* classes in UKW university in Bydgoszcz.
Here we're providing necessary tools to prepare data for transfer learning of an object detection model.

The main goal of this project is to create mechanism for recognizing vehicles license plates.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites
- Python 3.4+ _(tensorflow does not support python 3.8 yet)_
- Git - [instruction](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Protobuff - [instruction](https://developers.google.com/protocol-buffers/docs/downloads)
- Tensorflow Object Detection API - [instruction](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 
- Google Cloud account and tools _(for running training on cloud TPU)_ - [1](https://cloud.google.com/sdk/docs/quickstarts) and [2](https://cloud.google.com/storage/docs/gsutil_install)

Check `Python` installation

``` 
python3 --version
pip3 --version
virtualenv --version
```

Preferably create `Virtualenv` 

``` 
virtualenv --system-site-packages -p python3 ./venv
```

Activate `Virtualenv` 

``` 
source ./venv/bin/activate  # sh, bash, ksh, or zsh
```

Install reqiurements

```
pip install -r requirements.txt # This installs needed python packages
pip list # shows installed packages
```

To verify `tensorflow` installation use this command

``` 
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

To exit `Virtualenv` 

``` 
deactivate # don't exit until you're done using TensorFlow
```

### Instal Object Detection API

Install [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) - follow instructions in this repository.

**`Note!`** It still requires  tensorflow version _`1.x`_

Verify istallation:
```
python object_detection/builders/model_builder_test.py
```

Finally clone this repo:

```
git clone https://github.com/KonradKarimi/K9UVR_ML.git
```

***Now you ready to go!*** :smirk:

# Create dataset

Here's are instructions for preparing dataset for model training.

1. Prepare folder containing images of objects you want to detect.

2. Split your images into training and evaluation dataset. You can use [split-folders](https://pypi.org/project/split-folders/) for convenience. _(~70% images for training set)_

3. Use [RectLabel](https://rectlabel.com/) (Mac) or [VOTT](https://github.com/microsoft/VoTT#download-and-install-a-release-package-for-your-platform-recommended) (Windows / Linux) to annotate your files. You should create folder named `annotations` containing `.xml` annotations files inside folder with images.

4. Use `xml_to_csv.py`. **Important!** - check if `annotations` folder is present inside folder with images.

    `Ex.`
    ``` 
    python xml_to_csv.py --path_to_imgs="here path to images" --output_name="name your output file"
    ```

5. Use `generate_tfrecords.py` to create `.record` files for training and evaluation.
    
    `Ex.`
    ``` 
    #For training:
    python generate_tfrecords.py \
    --csv_input=data/train_labels.csv \
    --output_path=data/records/train.record  \
    --image_dir=/path_to_imgs_folder

    #For evaluation:
    python generate_tfrecords.py \
    --csv_input=data/test_labels.csv \
    --output_path=/data/records/test.record  \
    --image_dir=/path_to_imgs_folder
    ```
6. Create `label map` file. This is how it should look:

    **`Important`**, always start with index 1 because 0 is reserved. Save this file as `.pbtxt`

    `Ex.`
    ```
    item {
        id: 1
        name: 'license_plate'
    }
    ```

    **At this moment you should have two `.record` files which you use for training and evaluation of your model. Also you should have `.pbtxt` label map file.**

# Prepare object detection model

Here we're using method called `transfer learning`. Meaning that we need to download pre-trained model first. In this repo we're providing model downloaded from tensorflow's repository under ***models*** directory in this repo. You can go with that or you can download other supported model from the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Keep in mind to choose ***mobile*** version of model now.

1. Download pre-trained model from [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and unzip it.

2. Now download corresponding `.config` file from this [repository](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) (just copy content of chosen config file). Keep in mind that you're going to convert model  to `tflite` format later.


**At this moment you should have `.config` file for your detection model and `models` folder holding downloaded model whih contains `checkpoint` and multiple `model.ckpt.*` files.**

# Training the Model

Here's instructions on how you prepare training enviroment in Google Cloud Platform. Then you're going to run training job on GCP.

## Prepare Google Cloud Platform
1. Create project [here](http://console.cloud.google.com/).
2. Enable billing [how to docs](https://cloud.google.com/billing/docs/how-to/modify-project).
3. Enable APIs [here](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component&_ga=2.43515109.-1978295503.1509743045).
4. Install `gcloud` tool instructions [here](https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu).
5. Install `gsutil` tool instructions [here](https://cloud.google.com/storage/docs/gsutil_install).
6. Set your current project to the one you just created:
    ```
    gcloud config set project YOUR_PROJECT_NAME
    ```
7. Create a Cloud Storage bucket:

    ```
    gsutil mb gs://YOUR_UNIQUE_BUCKET_NAME
    ```
    - possibly could request you to use
        ```
        gcloud auth login
        ```
8. Set up two environment variables:
    ```
    export PROJECT="YOUR_PROJECT_ID"
    export GCS_BUCKET="YOUR_BUCKET_NAME"
    ```
9. Get the name of your service account:
    ```
    curl -H "Authorization: Bearer $(gcloud auth print-access-token)"  \
    https://ml.googleapis.com/v1/projects/${PROJECT}:getConfig
    ```
10. Copy the value of `tpuServiceAccount` should look:
    ```
    your-service-account-12345@cloud-tpu.iam.gserviceaccount.com
    ```
11. Add `tpuServiceAccount` to environment variable:
    ```
    export TPU_ACCOUNT=your-service-account
    ```
12. Grant the ml.serviceAgent role to your TPU service account:
    ```
    gcloud projects add-iam-policy-binding $PROJECT  \
    --member serviceAccount:$TPU_ACCOUNT --role roles/ml.serviceAgent
    ```

## Move files to storage bucket

Copy `.record` files to storage bucket:
```
gsutil -m cp -r data/records/*.record gs://${GCS_BUCKET}/data/
```
Then copy `.pbtxt` file:
```
gsutil cp data/labels/k9_detection_label_map.pbtxt gs://${GCS_BUCKET}/data/k9_detection_label_map.pbtxt
```
Copy downloaded model files:
```
gsutil cp models/ssd_mobilenet_v3_small_coco_2019_08_14/model.ckpt.* gs://${GCS_BUCKET}/data/
```

## Prepare model config

Edit `.config` file. Add following lines in `train_config:{ ... }` section:
```
fine_tune_checkpoint: "gs://your-bucket/data/model.ckpt"
fine_tune_checkpoint_type: "detection"
```

_Optionally add quantization at the end of the `.config`:_
```
graph_rewriter {
  quantization {
    delay: 1800
    activation_bits: 8
    weight_bits: 8
  }
}
```
_`"delay: 1800"`_ means it start to quantinize after 1800 step.

Point to your dataset files `.record` on storage bucket:
```
train_input_reader: {
  tf_record_input_reader {
    input_path: "gs://GCS_BUCKET/data/train.record"
  }
  label_map_path: "gs://GCS_BUCKET/data/k9_detection_label_map.pbtxt"
}


eval_input_reader: {
  tf_record_input_reader {
    input_path: "gs://GCS_BUCKET/data/val.record"
  }
  label_map_path: "gs://GCS_BUCKET/data/k9_detection_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
```

Then copy `.config` to storage bucket and rename it to `pipeline.config`:
```
gsutil cp config/ssdlite_mobilenet_v3_small_k9.config gs://${GCS_BUCKET}/data/pipeline.config
```

Now we need to pack and send to bucket packaged Object Detection API's files needed for training job to run. Go to folder where Object Detection API is installed and cd into _research/_ folder. ***Parentheses are part of the command!***
```
bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
python setup.py sdist
(cd slim && python setup.py sdist)
```
***There is no need for copying to storage bucket manually. We provide path to these files in command to run training/evaluation job on Goodgle Cloud***

## Running the training

To start training of the model, run the following command:
```
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
--job-dir=gs://${GCS_BUCKET}/train \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_tpu_main \
--runtime-version 1.15 \
--scale-tier BASIC_TPU \
--region us-central1 \
-- \
--model_dir=gs://${GCS_BUCKET}/train \
--tpu_zone us-central1 \
--pipeline_config_path=gs://${GCS_BUCKET}/data/pipeline.config
```

Then start evaluation immediately:
```
gcloud ml-engine jobs submit training `whoami`_object_detection_eval_validation_`date +%s` \
--job-dir=gs://${GCS_BUCKET}/train \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_main \
--runtime-version 1.15 \
--scale-tier BASIC_GPU \
--region us-central1 \
-- \
--model_dir=gs://${GCS_BUCKET}/train \
--pipeline_config_path=gs://${GCS_BUCKET}/data/pipeline.config \
--checkpoint_dir=gs://${GCS_BUCKET}/train
```

## Check training progress in tensorboard

To check how training is proceeding run tensorboard:
``` 
tensorboard --logdir=gs://${GCS_BUCKET}/train
```
**Tensorboard starts on localhost:6006**

# Getting model after training

Now when model training is ended we want to download model and then convert it to `.tflite` format to use on mobile devices.

## Getting Frozen graph
First we have to froze the model's graph. Run following command from `/research` folder.

```
python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```
 This will output **two** files in `$OUTPUT_DIR` folder.
 **`tflite_graph.pb`** and **`tflite_graph.pbtxt`**.

 **`Note!`** if you don't want to use `.tflite` format next you have to froze graph with ***`export_inference_graph.py!`*** not _`export_tflite_ssd_graph.py`_

 ## Converting model to `.tflite` format

 Now you can convert model for the wanted format. You can use provided script **`convert_to_tflite.py`** or use cli tool from tensorflow library.

 ```
 python convert_to_tflite.py --model=/tmp/tflite_graph.pb --output_name=detect.tflite
 ```

 **`Note!`** Script will export file to `converted_models/` folder.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/KonradKarimi/K9UVR_ML/tags).

## Authors

* **Konrad Karimi** - *Initial work* - [KonradKarimi](https://github.com/KonradKarimi)
* **Przemysław Tarnecki** - *Implementing on device detection* - [Isaac-mtg](https://github.com/Isaac-mtg)
* **Mariusz Frelke** - *Data Preparation*

## License

This project is licensed under the GNU GPLv3  License - see the [LICENSE](licenses/LICENSE.md) file for details.
Parts of this project depends on tensorflow code - see the [LICENSE](licenses/LICENSE-tf.md) for details.

## Acknowledgments

* Uses parts of code from tensorflow repo and [this gist](https://gist.github.com/wbrickner/efedf8ab0ce1705de1372c1e2f49dd98)
* **Using tensorflow 2.1 and 1.15 simultaneously in some cases**
* A bunch of typos included :laughing:

## Changelog

### 1.0.0
* Initial release

