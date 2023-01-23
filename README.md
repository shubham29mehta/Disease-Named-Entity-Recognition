# Named entity recognition for disease identification

The goal of this project/repository is to identify **diseases** in a text transformer models. Both bert base and biobert were fintuned on ncbi disease dataset

# Data Source

You can get the data from [here](https://huggingface.co/datasets/ncbi_disease).

# Evaluation metrics

Following are the results obtained using finetuned biobert. Fine tuned Biobert achieved 2.5% better F1 score as compared to finetuned bert (base)

* Precision: 83 %
* Recall: 89.7 %
* F1: 86.3 %
* Accuracy: 98.3 %

# Web Application

[Here](https://huggingface.co/spaces/shubham555/BioBERT_NER_Disease_Identification) is the link to Web Application deployed on HuggingFace Space using gradio.
