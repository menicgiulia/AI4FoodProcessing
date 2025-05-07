# Informatics for Food Processing 
#### Authors: Gordana Ispirova, Michael Sebek, Giulia Menichetti (giulia.menichetti@channing.harvard.edu)

## Introduction

This GitHub repository is for Chapter 11 `Informatics for Food Processing'' of the AgriFood Informatics book published by the Royal Society of Chemistry (in press). The chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning (ML), artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations—particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models (LLMs) like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research. This Github contains the data and scripts to reproduce the case study with the chapter as well as the models and metrics reported within the chapter.

## Case Study workflow
![Overview Pipeline](https://raw.githubusercontent.com/menicgiulia/AI4FoodProcessing/main/image/case_study_pipeline.png)

### Open Food Facts 
The Open Food Facts database, downloaded January 2025, is used as input data in all models. The data was filtered to ensure food items reported the ingredient list, number of additives, and the 11 most abundantly reported nutrients within Open Food Facts (proteins, fat, carbohydrates, sugars, fiber, calcium, iron, sodium, cholesterol, saturated fat, and trans fat). Since the study includes BERT and BioBERT models, LLMs trained on english, the dataset was additionally filtered to ensure there was an english ingredient list available.

- Explanatory and FoodProX Models use ```.../Data/Filtered_OFF.zip```
- LLM Model with BERT uses ```.../Data/bert_embeddings.zip```
- LLM Model with BioBERT uses ```.../Data/bio_bert_embeddings.zip```

The BERT and BioBERT embeddings were generated using the sentences found within ```.../Data/Filtered_OFF_with_sentences.zip```. Note that the model scripts only work with uncompressed data files, please unzip the data files prior to running the scripts. If you want to generate your own embeddings, the script used to curate the Open Food Facts dataset and generate the embeddings can be found at ```/Scripts/Dataset_formating_and_embeddings_generation.py```.

### Model Type
Each model within the case study uses select fields within the Open Food Facts dataset. The data is separated such that 20% of the data is used in hyperparameter tuning to find the optimal parameters for each model. The remaining 80% is used in the cross-validation during training of the models. These separations are set such that every model uses the same set for hyperparameter tuning and has the same splits within their cross-validation during training. The code to generate the splits for all models is found at ```.../Scripts/Make_splits_and_hyperparametertuning_set.py```. To use the same splits as in the book chapter:

- 20% hyperparameter tuning: ```.../Models/tuning_data_indexes.csv.zip```
- 80% training data: ```.../Models/training_splits.pkl.zip```

The codes for the various architectures used for each model are found below:

#### Explanatory Models
- Ingredient count with random forest: ```.../Scripts/Explanatory_model_num_of_ingredients.py```
- Additive count with random forest: ```.../Scripts/Explanatory_model_num_of_additives.py```

#### FoodProX Models
- FoodProX 11 nutrients with random forest: ```.../Scripts/FoodProX_model_11_nutrients.py```
- FoodProX 11 nutrients and additive count with random forest: ```.../Scripts/FoodProX_model_11_nutrients_and_additives.py```

#### LLM Models
- BERT with random forest: ```.../Scripts/BERT_Random_Forest_Classifier.py```
- BERT with XGBoost: ```.../Scripts/BERT_XGBoost_Classifier.py```
- BERT with neural network: ```.../Scripts/BERT_Neural_Network_Classifier.py```
- BioBERT with random forest: ```.../Scripts/BioBERT_Random_Forest_Classifier.py```
- BioBERT with XGBoost: ```.../Scripts/BioBERT_XGBoost_Classifier.py```
- BioBERT with neural network: ```.../Scripts/BioBERT_Neural_Network_Classifier.py```

#### Model Performances
We analyze the performance of each model using the code: ```.../Scripts/functions_for_evalution.py```. Each model script calls the functions from this code to calculate the AUC and AUP scores for each fold of the cross validation and for each NOVA classification. The average and standard deviation of the AUC and AUP scores are provided within the chapter and within ```.../Metrics```. The code also provides ROC and PRC curves for the models. All models used in the chapter are stored in ```.../Models```. The script to regenerate the figures within the chapter is found at ```/Scripts/Figures.py```.
