# New Category Discovery with Robust Pseudo Label Training and Source Domain Retraining


## Model Preparation
Get the pre-trained [BERT](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model and convert it into [Pytorch](https://huggingface.co/transformers/converting_tensorflow_models.html). 

Set the path of the uncased-bert model (parameter "bert_model" in init_parameter.py).

## Usage

Run the experiments by: 
```
sh run.sh
```
You can change the parameters in the script. 
