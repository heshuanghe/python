###  MLP
Build a Multi-Layer Perceptron.
- Create a new MLP with any given number of inputs, any number of outputs (can be sigmoidal or linear), and any number of hidden units (sigmoidal/tanh) in a single layer.
- Initialise the weights of the MLP to small random values.
- Predict the outputs corresponding to an input vector.
- Implement learning by backpropagation.

###  TextClassification
#### Overview 
The objective of this project is to scrape consumer reviews from a set of web pages and evaluate the performance of text classification on the data. The reviews have been divided into five categories here:
http://mlg.ucd.ie/modules/yalp
Each review has a star rating. We assume that 1-star to 3-star reviews are “negative”, and 4-star to 5-star reviews as “positive”.
#### Tasks
1) Select two review categories. Scrape all reviews for each category and store them as two separate datasets. For each review, store the review text and a class label (i.e. whether the review is “positive” or “negative”). 
2) From the reviews in this category, apply appropriate preprocessing steps to create a numeric representation of the data, suitable for classification.
Build a classification model using a classifier, to distinguish between “positive” and “negative” reviews.
Test the predictions of the classification model using an appropriate evaluation strategy. Report and discuss the evaluation results.
3) Evaluate how well the two classification models transfer between category. That is, run experiments to:
Train a classification model on the data from “Category A”, and evaluate its performance on the data from “Category B”.
Train a classification model on the data from “Category B”, and evaluate its performance on the data from “Category A”.


###  DataCollection&Preparation
#### Overview 
The objective of this project is to collect a dataset from one or more open web APIs, and use Python to preprocess and analyse the collected data.
#### Tasks
1) Choose at least one open web APIs as data source (not static datasets).
2) Collect data from your API(s) using Python. Depending on the API(s), you may need to repeat the collection process multiple times to download sufficient data.
3) Parse the collected data, and store it in an appropriate file format for subsequent analysis (e.g. plain text, JSON, XML, CSV).
4) Load and represent the data using an appropriate data structure. Apply any preprocessing steps that might be required to clean/filter/combine the data before analysis.
5) Analyse and summarise the cleaned dataset, using tables and visualisations where appropriate.

