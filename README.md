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
To access the code used to run analysis, refer to [Notebook.ipynb](/Notebook.ipynb).

For this analysis, we will use a Support Vector Machine Regression. The basic idea behind SVR is to find the best fit line, which is the hyperplane that has the maximum number of points.

**Why?** Was proven to have highest accuracy when compared to other supervised machine learning models according to [Kaur et al., 2019](https://www.mdpi.com/2076-3417/9/8/1613).

The Support Vector Machine methodology used was adapted from [Kaur et al, 2019](https://www.mdpi.com/2076-3417/9/8/1613), [SVM regression tutorial](https://github.com/AmirAli5/Machine-Learning/blob/main/Supervised%20Machine%20Learning/Regression/3.%20Support%20Vector%20Regression/Support%20Vector%20Regression.ipynb), and [SVM classifier tutorial](https://www.youtube.com/watch?v=8A7L0GsBiLQ). 

References used to write code: [source 1](https://www.youtube.com/watch?v=8A7L0GsBiLQ), [source 2](https://github.com/AmirAli5/Machine-Learning/blob/main/Supervised%20Machine%20Learning/Regression/3.%20Support%20Vector%20Regression/Support%20Vector%20Regression.ipynb), [source 3](https://www.youtube.com/watch?v=8A7L0GsBiLQ)

## Data cleaning and wrangling 

## **Q1:** Can we accurately predict this year's (2023) happiness?
**PART A:** Trained and tested our model on data between 2008-2022 to create a prediction for the happiness score for each country in the year 2023.

**PART B:** Compared predicted 2023 happiness to actual 2023 data.


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
![predicted2023_vs_actual2023](https://github.com/nooralteneiji/Supervised-Machine-Learning-on-Data-From-The-World-Happiness-Report/blob/main/Pipeline%20Outputs/Figures/predicted2023_vs_actual2023.png)image.






## Visualizations

# Limitations 

# Future directions 
* Conduct further regression analyses to understand the contribution of each variable to happiness while keeping others constant. Though we were able to visually see if two variables predicted happiness, for future directions we can construct a linear fixed effects model to get the weight each variable has towards predicting happines. This method will also inform us if there are more than just 2 interacting variables.
* Add governance-Quality Measures based on Data from the Worldwide Governance Indicators (WGI) Project.
* Build a Bayesian Network was adapted from [Chaudhary et al., 2020](https://arxiv.org/abs/2007.09181) to build a map of all the variables and see the strength each factor has on the predictor.

# Conclusion 
As a consequence, SVM regression might be used to determine the happiness of a country. If happiness can be predicted at an early stage, it is beneficial to stakeholders such that they can take preventative measures
and maximize life satisfaction in their respective countries.