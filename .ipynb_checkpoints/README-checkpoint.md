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
3. Split the pre-2023 and 2023 data into dependent and independent variables

## Feature engineering 


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