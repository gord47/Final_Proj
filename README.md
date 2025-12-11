# Classifying Hate Speech using DistilBeRT

The intent of this project is to provide a transformer based attempt to accurately classify hate speech texts using transformer based methods.

## Goal of the project
The goal of this project was to achieve a significant increase in performance over my chosen baseline (above 5%), which uses a simple transformer encoder model

## Setup instructions

If running on local machine, ensure the following are created
- Ensure python3 is installed on machine 
- create a virtual environment usint `python -m venv .venv`

### Installing requirements
#### Local Machine
1. activate the virtual environment: `.venv/Scripts/activate`
2. install dependencies: `pip install -r requirements.txt`

#### Google Colab
In the demo.ipynb script, insert a new code cell at the beginning of the file. Insert and run the following to install all required dependencies: 

`!pip install -r requirements.txt`

### Running the demo
To run the [demo script](demo.ipynb), under the checkpoints folder, the model checkpoint must be stored, which can be accessed in the following [link](https://drive.google.com/file/d/1cMaa07TLOE2kEw3s_Zn-EFUENKrlN9kV/view?usp=sharing)

This checkpoint file must be placed under a `checkpoints` folder to be accessed by the demo project.

### Expected results
When all cells have been executed in the demo script, a prediction file will be written under the results folder. This will contain:
- Text to classify
- Predicted class (hateful = 1, non hateful = 0)
- Actual class
- probability (Confidence model had when making the prediction. Closer to 0.5 means more unsure of prediction)
About 200 entries will be produced from the run of the demo script

## Data source
The source of this data is present [here](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data?select=HateSpeechDatasetBalanced.csv). Data is procured through kagglehub so no need for direct downloading of the dataset
