# Installation

## Download Prebuilt Binaries
You can download the latest release [here](https://github.com/Deepshift/DeepCreamPy/releases/latest) or find all previous releases [here](https://github.com/Deepshift/DeepCreamPy/releases).
Binary are available for Windows 64-bit, Linux-64-bit & MacOS aarch64

## Run Code Yourself
TODO: update

If you want to run the code yourself, you can clone this repo and download the model from https://drive.google.com/open?id=1IMwzqZUuRnTv5jcuKdvZx-RZweknww5x. Unzip the file into the /models/ folder.

If you want access to older models, see https://drive.google.com/open?id=1_A0xFeJhrqpmulA6cC-a7RxJoQOD2RKm.

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## running the code using Docker

Once the input images and model have been placed in `decensor_input` and `models` respectively,
the code can be run in the command line using docker (or podman), to avoid managing dependencies manually.

to build the container image use the command:
```
docker build -t deepcreampy .
```

then to desensor bar censors run the following command:
```
docker run --rm -v $(pwd)/models:/opt/DeepCreamPy/models -v $(pwd)/decensor_input:/opt/DeepCreamPy/decensor_input -v $(pwd)/decensor_output:/opt/DeepCreamPy/decensor_output deepcreampy
```

to desensor mosaics run the following command:
```
docker run --rm -v $(pwd)/models:/opt/DeepCreamPy/models -v $(pwd)/decensor_input:/opt/DeepCreamPy/decensor_input -v $(pwd)/decensor_input_original:/opt/DeepCreamPy/decensor_input_original -v $(pwd)/decensor_output:/opt/DeepCreamPy/decensor_output deepcreampy --is_mosaic=true
```

the contents of `decensor_input` and `decensor_input_original` are explained in the [decensoring tutorial](USAGE.md).

### Dependencies (for lib)
- Python 3.12
- TensorFlow 2
- Keras
- Pillow

### Dependencies (for app)
- fastapi
- uvicorn
- pydantic
- lib-cream-py

No GPU required!
