### The complete detail what I have been in this project

### Dataset Used
### Bank Marketing Data
    The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
    
    The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
    
Basically, this problem have four datasets with different number of features but I use the dataset named 'bank-additional-full.csv' beacuse it contained all the features which are collected for the purpose of this project.
 
The dataset contained 20 independent features (including numerical and categorical) and 1 dependent feature that have to predict it.
### Evaluation Metric Used is Accuracy and ROC_AUC_SCORE (Due to imbalanced class)
### The overall accuracy I got is 91.7% and ROC_AUC_SCORE is 76.02%
### Packages Used
    numpy
    pandas
    matplotlib
    seaborn
    scipy
    scikit-learn
    lightgbm
    xgboost
    catboost
    
### Files attached
    1. ReadME.md- Complete detail of the project
    2. Analysis_Modeling.ipynb- Complete analysis and modeling of the problem
    3. Executable.py- This file take input as data path and print out result as well
    4. predicted.csv- The prediction of rest 20% of the validating data

### Approach used to solve the problem
    - First and foremost have to read the data
    - Do some descriptive analysis like data description, information, etc.
    - Clean the Data
    
    #Now, actually what I did here
    1. Did univariate analysis for each features present
        And come to know that this data is imbalanced (the difference between the +ve and -ve result is very high)
        
    2. Did bivariate analysis using one independent feature and target feature (basically, here I used the violinplot, scatter plot and pairplot)
        And come to know the point how does the categorical features depends on the target feature
        
    3. Find the correlation between the features (continuous features)
        There have been lots of variation there some of them show highly +ve correlation on the other hand some show highly -ve correlation. In that case  I was in confusion and think about how to choose the important features from them
        
    4. Next step is data encoding
    
    5. Now again I find out the correlation between the dependent and independent features and plot it and here I come to know how actually the independent features are correlated with the dependent features (One interesting point I came to know that the some of lowest correlated features show the highest feature importance when I fit it to the model)
    
    6. Used chi2 statistical test to find out the important features
    
    7. Used f_classif statistical test to find out the important features
    
    8. Used ExtraTreeClassifier to find the features importances
    
    9. Used Recursive Feature Elimination (RFE) method to find out the best features at all (estimators used ExtraTreeClassifier)
        From the above two test I'm getting approximately same features as important features
        
    10. Did spot check model to find out the best model (metrics used AUC score)
        - First I used all the features for the spot checking 
            I got the LogisticRegression performed well in case of AUC score but LightGBM performed well in case of f1_score
        - 2nd I used the Kbest features of RFE
            Got the approx approx same result as above
        - 3rd I used the Kbest features of f_classif
            In that case I also got the approximately same
            
        From the above analysis I come to the point that why not I use all the features and from noow I used all the independent features beacuse the results are not differ more.
        
    11. Used the Catboost model as baseline and as tuned model
        Here, the baseline model performed well as comparison to the tuned one
        
    12. Use tuned LightGBM model 
        Now here, the LightGBM performed well as comparison to all the other model either in case of ROC_AUC_SCORE or f1_score
        
    (I split the data in 80:20 ratio [80% for training and 20% for testing])
        
        