# Experiments for the paper 'Attribution based Explanations cannot be robust and recourse sensitive'

This repository contains all the relevant code to run the experiments that were used in the paper
'Attribution based Explanations cannot be Robust and Recourse sensitive'. 

## Installing the requirements
It is advised to create a new environment and install all necessary dependencies. 
```
pip install -r requirements.txt
```

There can be an issue that the package `cv2` is not installed. To make sure that it is, run the 
following command:

```
pip install opencv-python
```

## Running the experiments
All experiments can be run by running the following script
```
python main.py 
```

### Options
If you do not want to run all experiments, but only a subset of them, you can pass a [y/n] argument 
indicating which experiments you want to run. For example
```
python main.py --gradient y --lime n --shap n
```
will only generate the pictures that use the gradient based explanations.


## Running the experiments with Docker

Alternatively, if you want to run the experiments using docker. You can build the image using
```
docker build -t recourse_experiments . 
```
To run the experiment you type
```
docker run \
    -v /path/to/save/results/pickled_data:/usr/src/pickled_data \
    -v /path/to/save/results/vanilla_gradients:/usr/src/vanilla_gradients \
    -v /path/to/save/results/smoothgrad:/usr/src/smoothgrad \
    -v /path/to/save/results/integrated_gradients:/usr/src/integrated_gradients \
    -v /path/to/save/results/lime_pictures:/usr/src/lime_pictures \
    -v /path/to/save/results/shap_pictures:/usr/src/shap_pictures \
    recourse_experiments:latest
```
