# Problem statement 

# Dataset
Task was to use World Happiness Report data from 2021. For this analysis, we also supplemented with happiness data from 2008-2023.

# Methodology
To access the code used to run analysis, refer to [Notebook.ipynb](/Notebook.ipynb)

## Data cleaning and wrangling 
The Support Vector Machine methodology used was adapted from [Kaur et al, 2019](https://www.mdpi.com/2076-3417/9/8/1613) and [this tutorial (though intended for building classification models, while ours is a regression)](https://www.youtube.com/watch?v=8A7L0GsBiLQ). 

**Why did we choose SVM?** Was proven to have highest accuracy when compared to other supervised machine learning models according to [Kaur et al., 2019](https://www.mdpi.com/2076-3417/9/8/1613).

Data cleaning/wrangling methods
Feature engineering
Modeling technique chosen
Measure of accuracy (if relevant)
Visualizations


# Limitations 

# Future directions 
* Conduct further regression analyses to understand the contribution of each variable to happiness while keeping others constant. Though we were able to visually see if two variables predicted happiness, for future directions we can construct a linear fixed effects model to get the weight each variable has towards predicting happines. This method will also inform us if there are more than just 2 interacting variables.
* Add governance-Quality Measures based on Data from the Worldwide Governance Indicators (WGI) Project.
* Build a Bayesian Network was adapted from [Chaudhary et al., 2020](https://arxiv.org/abs/2007.09181) to build a map of all the variables and see the strength each factor has on the predictor.

# Conclusion 
As a consequence, SVM regression might be used to determine the happiness of a country. If happiness can be predicted at an early stage, it is beneficial to stakeholders such that they can take preventative measures
and maximize life satisfaction in their respective countries.