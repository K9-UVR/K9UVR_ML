# K9 License Plate Recognition

Project created as an graduate team project.
Allows recognise unmarked vehicle's license plates.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

#### Prerequisites
- Python 3.4+ _(tensorflow does not support python 3.8 yet)_
- Git
- Protobuff
- Tensorflow Object Detection API
- Google Cloud account _(for learning on cloud TPU)_

Install **Python** >= 3.4 & <= 3.7

Check `Python` installation

``` 
python3 --version
pip3 --version
virtualenv --version
```

#### Preferably create `Virtualenv` 

``` 
virtualenv --system-site-packages -p python3 ./venv
```

Activate `Virtualenv` 

``` 
source ./venv/bin/activate  # sh, bash, ksh, or zsh
```

Install reqiurements

``` 
pip install --upgrage pip
pip install -r requirements.txt
pip list # shows installed packages
```

This should install all needed python packages.

To verify `tensofrlow` installation use this command

``` 
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

To exit `Virtualenv` 

``` 
deactivate # don't exit until you're done using TensorFlow
```

### Instal Object Detection API

Install [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 

**Note** It still depends on tensorflow _version 1_. 

Use:
```
pip install tensorflow==1.15
```

Test istallation:
```
python object_detection/builders/model_builder_test.py
```

***Now you ready to go!***

## Create datasets

Here you prepare Dataset for model training.

1. Prepare folder with images of objects you want to detect.

2. Split your images into training and evaluation dataset. You can use [split-folders](https://pypi.org/project/split-folders/) for convienience. _(~70%  images for training set)_

3. Use [RectLabel](https://rectlabel.com/) (Mac) or [VOTT](https://github.com/microsoft/VoTT#download-and-install-a-release-package-for-your-platform-recommended) (Windows / Linux) to annotate your files. You should have `annotations` folder containing `.xml` files inside folder with images.

4. Use `xml_to_csv.py`. **Important!** - check if `annotations` folder is present inside folder with images.

    `Ex.`
    ``` 
    python xml_to_csv.py --path_to_imgs="<here path to images>" --output_name="<name your output file>"
    ```

5. Use `generate_tfrecords.py` to create `.record` files for training and evaluation.
    
    `Ex.`
    ``` 
    For training:
    python generate_tfrecords.py \
    --csv_input=data/train_labels.csv \
    --output_path=train.record --image_dir=/Datasets/

    For evaluation:
    python generate_tfrecords.py \
    --csv_input=data/test_labels.csv \
    --output_path=test.record --image_dir=/Datasets/
    
    ```
6. Create `label map` file. This is how it should look:
    
    `Ex.`
    ```
    item {
        id: 1
        name: 'license_plate'
    }
    ```
    **`Important`**, always start with index 1 because 0 is reserved. Saved this file as `.pbtxt` format.

    **At this moment you should have two `.record` files which you use for training your model. Also you should have `.pbtxt` label map file**

## Prepare object detection model

Now you need to obtain config file for your object detection model.

1. Download `.config` file from this [repository](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) (just copy content of chosen config file). Keep in mind that you're going to convert model  to `tflite` format later. Please choose ***mobile*** versions of model now.
2. Download pre-trained model from [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and unzip it.

**At this moment you should have `.config` file for your detection model and folder holding downloaded model whih contains `checkpoint` and multiple `model.ckpt.*` files. You're going to use them for transfer learning.**

## Model Training

Here you prepare training enviroment in Google Cloud Platform. Then you're going to run model training job on GCP.

### Prepare Google Cloud Platform
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

### Move files to storage bucket

Copy `.record` files to storage bucket:
```
gsutil -m cp -r /data/*.record gs://${GCS_BUCKET}/data/
```
Then copy `.pbtxt` file:
```
gsutil cp k9_detection_label_map.pbtxt gs://${GCS_BUCKET}/data/k9_detection_label_map.pbtxt
```
Copy downloaded model files:
```
gsutil cp /models/ssd_mobilenet_v3_small_coco_2019_08_14/model.ckpt.* gs://${GCS_BUCKET}/data/
```

### Prepare model config

Edit `.config` file. Add following lines in `train_config:{ ... }` section:
```
fine_tune_checkpoint: "gs://your-bucket/data/model.ckpt"
fine_tune_checkpoint_type: "detection"
```

Optionally add quantization at the end of the `.config`:
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

Now we need to pack and send to bucket packaged Object Detection API. Go to folder where it's installed and cd into _research/_ folder. ***Parentheses are part of the command!***
```
bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
python setup.py sdist
(cd slim && python setup.py sdist)
```

### Running the training

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

Then start evaluation:
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

### Check progress in tensorboard

To check how training is proceeding run tensorboard:
``` 
tensorboard --logdir=gs://${GCS_BUCKET}/train
```
**Tensorboard starts on localhost:6006**

### And coding style tests

Explain what these tests test and why

``` 
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Konrad Karimi** - *Initial work* - [KonradKarimi](https://github.com/KonradKarimi)
* **Przemysław Tarnecki** - *Tests* - 
* **Mariusz Frelke** - *Data Preparations* - 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
* **FOR TRAINING NEED TO DOWNGRADE TO TFv1**

