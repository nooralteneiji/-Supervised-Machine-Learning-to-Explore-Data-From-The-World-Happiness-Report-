If a government can predict its citizens' happiness accurately, it can use that information to evaluate the success of its current policies and make adjustments where necessary. 

The objective of this project is to create a predictive model of happiness. By accurately predicting happiness, we can gain insights into the factors that contribute to well-being and inform policymakers about the effectiveness of their policies. Our model aims to provide a valuable tool for understanding and predicting happiness. 

# Dataset
Task was to use World Happiness Report data from 2021. For this analysis, we also supplemented with happiness data from 2008-2023.

Here are brief definitions for each variable from the World Happiness Report:

- `country`: The name of the country for which the happiness data is reported.
- `year`: The year in which the happiness data is recorded.
- `GDP`: Gross Domestic Product per capita, which measures the economic output per person in a country.
- `socialSupport`: The level of social support and the presence of a support network in a person's life.
- `lifeExpectancy`: The average number of years a person is expected to live in good health, capturing the overall health and well-being of individuals in a country.
- `freedom`: The degree of political and social freedoms enjoyed by individuals in a country.
- `generosity`: The measure of generosity, including charitable donations and helping others, in a country.
- `corruption`: The perceived level of corruption and trustworthiness within the public sector of a country.

# Methodology
To access the code used to run analysis, refer to [Notebook.ipynb](/Notebook.ipynb).

To create our predictive model, we employed a Support Vector Machine Regression (SVR) approach. SVR aims to find the best fit line, or hyperplane, that maximizes the number of data points. 

**Why?** Was proven to have highest accuracy when compared to other supervised machine learning models according to [Kaur et al., 2019](https://www.mdpi.com/2076-3417/9/8/1613).

The Support Vector Machine methodology used was adapted from [Kaur et al, 2019](https://www.mdpi.com/2076-3417/9/8/1613), [SVM regression tutorial](https://github.com/AmirAli5/Machine-Learning/blob/main/Supervised%20Machine%20Learning/Regression/3.%20Support%20Vector%20Regression/Support%20Vector%20Regression.ipynb), and [SVM classifier tutorial](https://www.youtube.com/watch?v=8A7L0GsBiLQ). 

References used to write code: [source 1](https://www.youtube.com/watch?v=8A7L0GsBiLQ), [source 2](https://github.com/AmirAli5/Machine-Learning/blob/main/Supervised%20Machine%20Learning/Regression/3.%20Support%20Vector%20Regression/Support%20Vector%20Regression.ipynb), [source 3](https://www.youtube.com/watch?v=8A7L0GsBiLQ)

## Data cleaning and wrangling 
1. Import the raw data from pre-2023 and 2023 datasets.
2. Remove columns that are not present in both datasets.
3. Standardize the naming scheme for dimensions across datasets.
4. Merge the pre-2023 and 2023 datasets into one dataframe.
5. Check for null or missing values in the dataframe.
6. Replace zero values with NaN to prepare for interpolation.
7. Use linear interpolation to estimate and replace missing values.

Overall, the workflow involves importing, cleaning, standardizing, and merging the data, handling missing values through interpolation, and analyzing the coverage of years and missing data for each country.

## **Q1:** Can we accurately predict this year's (2023) happiness?
1. Trained and tested our model on data between 2008-2022 to create a prediction for the happiness score for each country in 2023.
2. Compared predicted 2023 happiness to actual 2023 data.
3. Evaluated the model's performance using metrics such as R-squared, RMSE, and Accuracy.


### Data pre-processing
1. Created 2 dataframes; pre-2023 and 2023 data. 
2. Countries that were not present in the actual 2023 data were removed to ensure fair comparision later between actual and predicted 2023 scores.

### **Data splitting:** `X` and `y` (`pre2023` and `2023`) 
1. Split the pre-2023 and 2023 data into dependent and independent variables

### **Feature engineering:** One-hot encoding
**Purpose:** Ensure X is numeric to prep for scaling and inserting into SVM  
In order to scale our train and test set for `pre2023`, and feed the data into the SVM, all X variables need to be numeric. In our case `country` is categorical. As consequence, we need to use one-hot encoding to convert country data into a binary. Doing this process before training and splitting the `pre2023` data will not lead to data leakage. This is acceptable as even though for the `pre2023` dataframe  some countries have data for specific years, but not others, the entire dataset has the exact same countries represented in the `2023` data.

**How it works:** One-hot encoding will create a binary column for each category in the feature. The only downside is that the dimensionality of out data will increase since we have 195 countries in the world. 

**Why not ordinal encoding?** Even though we could maintain dimensionality by using ordinal encoding, we will not use it as it would assign a value to each country, implying a ranking--which is not true in our case. It would, for an example, cluster country 193, 194, 195 together because it would assume they are similiar.

### **Data splitting:**  Creating train and test set for `pre2023`
**How will we split the df?**
We will use the KFold function to generate the indices (AKA the position) to split `pre2023` into the training and testing sets. 

**What is it?**
This is a re-sampling procedure. It has a single parameter called 'k' that refers to the number of groups (or "folds") that `pre2023` will be split into. In our case, `pre2023` will be split into multiple different training and testing sets

**Why are we splitting `pre2023` into k=10 folds?*** 
It means it matters less how the data gets divided - every data point gets to be in a test set exactly once, and gets to be in a training set k-1 times.

**How does it work?** 
When we use the KFold function, a pair of indices is returned where one part of the indices refers to the training set and the other part to the test set. The indices themselves are integer values that correspond to the position of each data point in `pre2023`. 

We will later retrieve these indices to evaluate our model. 


### Scaling train and test set for `pre2023`
After making all X variables numeric, we can create train and test sets from `pre2023`, then we can scale the data. Scaling NEEDS to happen *after* the train and test split, otherwise the transformation will be done in accordance to the entire dataset, whereas we need the scaling to be done in accordance to the training and testing sets seperately. 

The resulting scaled training and testing sets for each fold are then appended to their respective lists.


### Creating the `pre2023` model
Earlier we split `pre2023` into 10 folds. Now we can use these folds to train the model 10 times, each time using a different subset as the testing data and the remaining data as the training data.

![Schematic illustrating how k-fold validation works](https://trituenhantao.io/wp-content/uploads/2020/01/k-fold.png)

**Why do we use k-fold validation?**
To provide a robust evaluation of the model performance. The average performance over the 10 trials gives us a better estimate of the model's true performance than a single train/test split. This means we can avoid overfitting where the model performs well on training data (learns the outliers/noise), but poorly on unseen data.

**How did we create the model?**
```
pre2023_model = SVR()
```
1. We use a `for` loop to iterates over all 10 folds we created earlier. 
2. We retrieve the corresponding training and test data (both the features X and the targets y) for the current fold from the pre-stored lists.
```
    # retrieve the data for this fold
    X_train = X_train_list[i]
    X_test = X_test_list[i]
    y_train = y_train_list[i]
    y_test = y_test_list[i]
    
```

3. In each iteration, the model is fitted to the training data in `X_train_list` and `y_train_list` for the current fold.
```
    pre2023_model.fit(X_train, y_train) # training 
```

4. The fitted model is used to make predictions on the test data for the current fold.
```
    y_pred = pre2023_model.predict(X_test) # creating prediction
    
```

5. We evaluate the model on each of the testing sets in `X_test_list` and `y_test_list` in each fold.

```
    # Calculate and store the performance metrics
    r2_scores.append(r2_score(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    # computing accuracy
    relative_error = np.abs((y_test - y_pred) / y_test) # Compute the relative error
    correct_predictions = np.sum(relative_error <= threshold) # Count the number of predictions that fall within the threshold
    accuracy_scores.append(correct_predictions / len(y_test)) # Compute and store the accuracy for this fold

```
6. After the loop, we calculate and print the average performance metrics across all folds to evaluate the overall performance of the model.

**How did our model built on `pre2023` perform?**
 R^2 = `0.889`, RMSE = `0.372`, Accuracy = `85.1%`

Where accuracy is defined as: If the difference between the predicted and actual values falls within this threshold, then the prediction is deemed correct.

These metrics give us a measure of how well your model is expected to perform on unseen data.

### Creating the `pre2023` model: Optimizing parameters 
Though our model perfromed quite well, let's find the optimal parameters to see if we can improve performance.

Since we have more than one parameter to optimize, we used the GridSearchCV() to test all possible combinations.

It turns out this is our ideal SVM regression model:
```
model = SVR(kernel='rbf', C = 100, epsilon = 0.1, gamma = 'auto')
```
**Result**
The SVM preformed really well straight out of the box, with only slight improvements being added to the R^2 and RSME after optimization, but no changce in accuracy.

Without tuning:  R^2 = `0.889`, RMSE = `0.372`, Accuracy = `85.1%`

Aftering tuning: R^2 = `0.892`, RMSE = `0.364`, Accuracy =  `85.1%`

### Evaluating the `pre2023` model: Compare to actual 2023 scores 
![predicted2023_vs_actual2023](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/predicted2023_vs_actual2023.png)
* The points fall along and cluster arounf the line of equality = precision.
* There is no systematic pattern to the points deviating from the line of equality (for example, if points for lower actual values are consistently overpredicted), this could indicate that the model does not have a bias / captures the relationship between the predictors and the outcome variable.

![residuals](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/residuals.png)

![dist residuals](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/Distributionresiduals.png)

### Final thoughts on 2023 model 
Since our model performed so well, it is reasonable to try an use to forecast the upcoming year.



## **Q2:** Can we accurately predict next year's (2023) happiness?

Where as in the 2023 model, the training and test dataset did not include 2023 data. In this model, 2023 data is included too.

We trained our model on data from 2005-2023, including 2023 data. This allowed us to make predictions for 2024 based on the trained model. We assessed the model's performance using metrics such as R-squared, RMSE, and Accuracy.

**A. Prepare to train model on `2023` (data from 2005-2023)**

**C. Evaluate 2023 model** 
   R^2 = `0.898`, RMSE = `0.358`, Accuracy =  `86.3%`
   

After training our Support Vector Machine model on the data from 2005-2023, we will use this trained model to forecast the 'happiness' score (y) for the year 2024.

Our forecasting approach is to use 2023 data as the input for predicting 2024 outcome.

The workflow for this methodology can be divided into the following steps:

1. Data Preparation: 2008-2023 train/test
   - Separate the target variable (happiness) from the features by dropping it from the copy of the dataframe.
   - Encode categorical variables using one-hot encoding.
   - Scale the training data (`X_2008_2023_encoded`) using the StandardScaler =

3. Data Preparation: X var for predicted 2024
    - Extract the data for the year 2023 from the original dataframe and drop the target variable.
    - Encode the categorical variables in the 2024 data using the same encoding as the training data.
    - Reorder the columns in the 2024 data to match the order in the training data.
    - Scale the 2024 data using the same scaler object used on the training data.

4. Model Creation and Training:
   - Create a Support Vector Regression (SVR) model with a radial basis function kernel and specified hyperparameters (C, epsilon, gamma).
   - Fit the model on the scaled training data.

5. Prediction for 2024:
   - Make predictions for the year 2024 using the trained model and the scaled 2024 data.

6. Cross-validation:
   - Prepare empty lists to store the performance metrics (R-squared, RMSE, Accuracy) for each fold.
   - Create a KFold object with the desired number of folds.
   - Perform cross-validation by splitting the scaled training data into train and test sets for each fold.
   - Create a new SVR model for each fold, fit it on the training data, and make predictions on the test data.
   - Calculate and store the performance metrics (R-squared, RMSE, Accuracy) for each fold.

7. Calculate Average Performance Metrics:
   - Calculate the average performance metrics (R-squared, RMSE) across all folds.

8. Bootstrap Resampling:
Since SVMs are not probailistic models, they do not inherently provide a way for us to calculate confidence intervals (unlike e.g., linear regression). However, we can use a rough approach to generate confidence intervals for our model through **bootstrapping**. 

**What does bootstrapping do?**
Using bootstrapping will allow us to draw samples from the dataset we used to predict 2024 data. It then replaces the taken samples to fit the model again and again, giving us a distribution of the estimates. With this distribution, we can take it's percentiles to create confidence intervals.

**How to interpret Cl in context of SCM**
In SVMs, these intervals do not necessarily capture the model's uncertainty about its predictions (unlike models that assume a normal distribution e.g., bayesian model), but rather the variability in the predictions that arises when re-fitting the model to different samples of the data.

A 95% confidence interval, in our case, means that if we were to repeat the sampling process many times, we would expect the true value to fall within this interval 95% of the time. However, this interpretation doesn't imply that there's a 95% chance the true value is within the interval; rather, it's a statement about the method's reliability if the process was repeated many times.

**How did we bootstrap?**
   - Initialize the SVR model.
   - Specify the number of bootstrap samples to create.
   - Fit and evaluate the model for each fold using bootstrap resampling:
     - Create a bootstrap sample by resampling the training data.
     - Fit a new SVR model on the bootstrap sample.
     - Predict the 2024 data using the model trained on the bootstrap sample.
     - Store the predictions from each bootstrap sample locally so that we don't need to run code again, even if kernel reset.

9. Calculate Confidence Intervals:
   - Calculate the lower and upper percentiles (2.5% and 97.5%) of the predictions obtained from bootstrap resampling.
   - Save the confidence intervals to a CSV file and a numpy file.

10. Visualization:

![predected vs 2023](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/happiness_forecast.png)

11. Calculate Percent Change:
    - Create a new DataFrame for 2023 and 2024 happiness values.
    - Merge the two DataFrames based on the country.
    - Add a percent change column to calculate the percentage change in happiness from 2023 to 2024.
    - Sort the DataFrame based on the percent change.

12. Visualization of Percentage Change:
    - Select the top positive percentage changes and the largest negative percentage changes.
    - Sort the negative percentage change DataFrame from smallest to largest.
    - Add a "Change" column to indicate positive or negative change.
    - Concatenate the positive and negative DataFrames.
    - Plot a bar chart showing the countries with the highest percent changes in happiness from 2023 to 2024.
   
![top prcnt change](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/happiness_forecast_percentage_change.png)   


### Final thoughts on 2024 model 
Hard to predict the future! The predictions are essentially a replication of 2023's data with some minor alterations due to the model. They don't account for potential new information or significant changes that may occur in 2024.

We won't be able to validate these predictions against actual values as we did for the previous years, because the real 2024 data is not available yet.

If there are large year-to-year fluctuations in happiness scores or the predictors, the model might not produce accurate predictions for 2024.


## **Q3:** Which variables are most strongly associated with happiness?
We explored the variables' importance in predicting happiness using permutation importance. Additionally, we investigated potential interactions between variables to identify which combinations had a significant impact on happiness.
### Which variables have the largest influence on determining happiness?
1. Permutation importance is computed for each variable in the dataset.
2. The average importance across multiple folds is calculated.
3. The sorted importances are used to create a horizontal bar plot.

![factors](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/factors_predict_happiness.png)


### Are there any interacting variables that predict happiness?
Here is a summary of the workflow without referring to variables:

1. Create pair plots to visualize potential interactions between features.

![pairplots](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/pairplot.png)
   
2. Compute the R-squared value for each pair of features to determine their predictive power.
3. Identify the pairs of features with moderate or above moderate correlation as potential predictors of happiness.
4. Create interactive and static 3D scatter plots for each pair of features and happiness.

![GDP_Life_Social](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/GDP_Life_Social.png)

![Happiness_predictedBy_Life_gdp](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/Happiness_predictedBy_Life_gdp.png)

![Happiness_predictedBy_gdp_Social](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/Happiness_predictedBy_gdp_Social.png)

![Happiness_predictedBy_life_Social](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/Happiness_predictedBy_life_Social.png)
   
# Limitations and Future directions 
* reliance on self-reported data
* the assumption that past trends will continue into the future. 
* Conduct further regression analyses to understand the contribution of each variable to happiness while keeping others constant. Though we were able to visually see if two variables predicted happiness, for future directions we can construct a linear fixed effects model to get the weight each variable has towards predicting happines. This method will also inform us if there are more than just 2 interacting variables.
* Add governance-Quality Measures based on Data from the Worldwide Governance Indicators (WGI) Project.
* Build a Bayesian Network was adapted from [Chaudhary et al., 2020](https://arxiv.org/abs/2007.09181) to build a map of all the variables and see the strength each factor has on the predictor.

# Conclusion 
As a consequence, SVM regression might be used to determine the happiness of a country. If happiness can be predicted at an early stage, it is beneficial to stakeholders such that they can take preventative measures
and maximize life satisfaction in their respective countries.