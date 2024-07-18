# A2Perf Benchmark Submission

This repository contains a template script for training and inference of a model for the A2Perf Benchmark Submission.

## Files

### 1. `inference.py`

#### Purpose
This script will handle the inference part of the submission. Fill in the functions required to load the trained model, preprocess observation data, and perform inference to predict actions.

#### Functions

- **`load_model(env)`**
  - **Description**: This function is intended to load and return the model.
  - **Arguments**:
    - `env`: The "env" parameter can be used to specify the environment or any other context for loading the model.

- **`infer_once(model, observation)`**
  - **Description**: Uses the loaded model (*assumed to be from Stable Baselines*) to run inference on a single observation. The function receives a "model" along with an "observation", and then returns the predicted action.
  - **Arguments**:
    - `model`: The model to use for inference.
    - `observation`: The input observation data.
  - **Returns**: The predicted action.

- **`preprocess_observation(observation)`**
  - **Description**: Preprocess the observation data before feeding it to the model. Modify this function to suit the preprocessing needs of your specific model.
  - **Arguments**:
    - `observation`: The raw observation data.
  - **Returns**: The preprocessed observation data.

- **`main(_)`**
  - **Description**: A placeholder main function required by the absl application.
  - **Arguments**:
    - `_`: Unused.

### 2. `train.py`

#### Purpose
This script will handle the training process of the model. Fill in the functions required for the training logic of the model, running the training loop, and saving the trained model.

#### Functions

- **`train()`**
  - **Description**: Encapsulates the entire training process, including model setup, training loop, and saving the trained model.

- **`main(_)`**
  - **Description**: The main function that initiates the training process when the script is executed.
  - **Arguments**:
    - `_`: Unused.