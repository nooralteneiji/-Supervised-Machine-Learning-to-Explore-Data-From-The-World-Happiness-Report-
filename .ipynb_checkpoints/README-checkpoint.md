# Problem statement 

# Dataset
Task was to use World Happiness Report data from 2021. For this analysis, we also supplemented with happiness data from 2008-2023.

## Variable definitions 
`country` 
`year`
`GDP`
`socialSupport`
`lifeExpectancy`
`freedom`
`generosity`
`corruption`

# Methodology
To access the code used to run analysis, refer to [Notebook.ipynb](/Notebook.ipynb)

## Data cleaning and wrangling 

## **Q1:** Can we accurately predict this year's (2023) happiness?
**PART A:** Trained and tested our model on data between 2008-2022 to create a prediction for the happiness score for each country in the year 2023. 
**PART B:** Compared predicted 2023 happiness to actual 2023 data.


### Method
## Data pre-processing
1. Created 2 dataframes; pre-2023 and 2023 data. 
2. Countries that were not present in the actual 2023 data were removed to ensure fair comparision later between actual and predicted 2023 scores.

## **Data splitting:** `X` and `y` (`pre2023` and `2023`) 
1. Split the pre-2023 and 2023 data into dependent and independent variables

## Feature engineering: One-hot encoding
**Purpose:** Ensure X is numeric to prep for scaling  
In order to scale our train and test set for `pre2023`, and feed the data into the SVM, all X variables need to be numeric. In our case `country` is categorical. As consequence, we need to use one-hot encoding to convert country data into a binary. Doing this process before training and splitting the `pre2023` data will not lead to data leakage. This is acceptable as even though for the `pre2023` dataframe  some countries have data for specific years, but not others, the entire dataset has the exact same countries represented in the `2023` data.

**How it works:** One-hot encoding will create a binary column for each category in the feature. The only downside is that the dimensionality of out data will increase since we have 195 countries in the world. 

**Why not ordinal encoding?** Even though we could maintain dimensionality by using ordinal encoding, we will not use it as it would assign a value to each country, implying a ranking--which is not true in our case. It would, for an example, cluster country 193, 194, 195 together because it would assume they are similiar.
## **Data splitting:** training and test set for `pre2023` 

## Scaling train and test set for `pre2023`
After making all X variables numeric, we can create train and test sets from `pre2023`, then we can scale the data. Scaling NEEDS to happen *after* the train and test split, otherwise the transformation will be done in accordance to the entire dataset, whereas we need the scaling to be done in accordance to the training and testing sets seperately. 
 

## Model selection 
For this analysis, we will use a Support Vector Machine Regression. The basic idea behind SVR is to find the best fit line, which is the hyperplane that has the maximum number of points.

**Why?** Was proven to have highest accuracy when compared to other supervised machine learning models according to [Kaur et al., 2019](https://www.mdpi.com/2076-3417/9/8/1613).

The Support Vector Machine methodology used was adapted from [Kaur et al, 2019](https://www.mdpi.com/2076-3417/9/8/1613), [SVM regression tutorial](https://github.com/AmirAli5/Machine-Learning/blob/main/Supervised%20Machine%20Learning/Regression/3.%20Support%20Vector%20Regression/Support%20Vector%20Regression.ipynb), and [SVM classifier tutorial](https://www.youtube.com/watch?v=8A7L0GsBiLQ). 

## Measuring accuracy 

References used to write code: [source 1](https://github.com/AmirAli5/Machine-Learning/blob/main/Supervised%20Machine%20Learning/Regression/3.%20Support%20Vector%20Regression/Support%20Vector%20Regression.ipynb)

## Visualizations

# Limitations 

# Future directions 
* Conduct further regression analyses to understand the contribution of each variable to happiness while keeping others constant. Though we were able to visually see if two variables predicted happiness, for future directions we can construct a linear fixed effects model to get the weight each variable has towards predicting happines. This method will also inform us if there are more than just 2 interacting variables.
* Add governance-Quality Measures based on Data from the Worldwide Governance Indicators (WGI) Project.
* Build a Bayesian Network was adapted from [Chaudhary et al., 2020](https://arxiv.org/abs/2007.09181) to build a map of all the variables and see the strength each factor has on the predictor.

# Conclusion 
As a consequence, SVM regression might be used to determine the happiness of a country. If happiness can be predicted at an early stage, it is beneficial to stakeholders such that they can take preventative measures
and maximize life satisfaction in their respective countries.