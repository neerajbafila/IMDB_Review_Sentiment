# To create env

# STEPS

## Step 01 - Create a repository by using template repository
## step 02 - Clone the new repository
## step 03 - Create a conda environment after opening the repository in VSCODE
```
conda create --prefix ./env python=3.10 -y
```
## activate environment
```
conda activate ./env
```
### or
```
source activate ./env
```

## STEP 04- install the requirements
```
pip install -r requirements.txt
```

## step 05- install tensorFlow with Cuda

```
pip install  tensorflow==2.10.1

```
step 06- install setup.py if -e . not mentioned in requirements.txt
```
pip install -e .
```
## step 07- export conda environment in yaml file
```
conda env export > conda.yaml 
```
## To run ml project from local

```
mlflow run . --env-manager local
```