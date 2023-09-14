# ECS171



## Introduction
*A completed write that includes the following
Introduction of your project. Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?
Figures (of your choosing to help with the narration of your story) with legends (similar to a scientific paper) For reference you search machine learning and your model in google scholar for reference examples.*

&nbsp;&nbsp;&nbsp; Originally we had a dataset containing 10,000 movies that we wanted to use based on budget and revenue. However, after removing the zero values, we were left with around 4,000 usable samples. After replacing the data with mean imputation, we noticed our model was very inaccurate and decided to change our data set. 

&nbsp;&nbsp;&nbsp; The project we decided on was to create predictive models on the electricity market in New South Wales, Australia. The dataset also contained information on the neighboring state, Victoria, in order to alleviate variance. This data contained over 45,000 samples acquired from May 7th 1996 to December 5th 1998. This market had features that contained fluctuating prices that were influenced by the supply and demand of electricity. We chose this dataset for its unique nature and potential insights it could provide into the dynamics of electricity markets. The fluctuating prices and the influencing factors as well as the dataset's size and complexity make this an interesting and challenging dataset to analyze. That is why machine learning and predictive modeling is great for this data.

&nbsp;&nbsp;&nbsp;Predictive models in any market allow companies to anticipate changes in demand and adjust their output and production for a more efficient operation. Having predicitive models can help consumers plan their usage on a product and ultimately save money. A predictive model on the electricity market can be help stabilize the electricity grid and further help balance the supply and demand. 


## Methods section 
*(this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods*

### Preprocessing:

#### Original Movie set Data
 We are changing the date feature by omitting the dashes in the 'date' data inputs and then converting them into integers, that way we change them all into 8 digit int variables. We will also change the genre feature from an array of strings into a codefied array of int variables. It will remain an array since the movies are classified under multiple categories, but all the genres seem to repeat, so they will be easily codefied into ints. Same goes for the Production companies, they are currently array's of strings listing the production companies involved in the making of the movie, however we will just convert the string values into int values. We will drop the title column as well as the tagline column since there is no way of codefying them in a repeatable manner. For the budget column and revenue, there are about 80% of data points missing (i.e. they are listed as zero), so it is probably not possible to omitt these, so instead we probably will just ignore these columns but keep the datapoints in.

We decided to drop the tagline, title, and overview columns, and all of the nan values. We also replaced the 0's with the mean of their respective columns. We ran linear regression on the data and the error was massive, unsurprisingly. The model was trained on budget to predit revenue and likely resulted in underfitting because of the mean replacement technique we used.

#### New Electricity Data Set
&nbsp;&nbsp;&nbsp; The `class` feature had either values as `b'UP'` or `b'DOWN'`. Having the byte prefixed to the string made it difficult to classify and therefore was dropped via the code ```df['class'] = df['class'].str.decode('utf-8')```. We then replaced the value `UP` and `DOWN` with the numeric values `1` and `0` respectively. We then dropped null values with `df.dropna(inplace=True)`. The day label had values 1-7 prepended with the byte character. We changed the values to their respective numeric value with the map method `df['day'] = df['day'].map({b'1': 1, b'2': 2, b'3': 3, b'4': 4, b'5': 5, b'6': 6, b'7': 7})`. Most of the the data came normalized and therefore was not a required step to implement. 
### Data Exploration:
Using the pairplot method from the seaborn library, we observed a positive correlation between the features `nswdemand` and `vicdemand` as well as a saturation effect between `nswprice` and `nswdemand`. There is also a negative correlation between `transfer` and both demands (`nswdemand`, `vicdemand`) 
### Models Chosen
#### 1. Polynomial Regression
We use the `PolynomialFeatures` from the sklearn lib and generate a feature matrix of all polynomial combinations. We use a pair of features with the degree paraemter set to 2. The model if fit with the polynomial features using `LinearRegression` from the sklearn lib. we use the `predict` method to plot the regression line with points of the real data. 
##### Parameters chosen
`features = ['day', 'nswprice', 'date']`
`target = [nswdemand]`
`degree = 2`
#### 2. Logistic Regression
We set the target and features and split the data where 20% of the data will be used for testing and 80% used for training. We use `LogisticRegression` from the sklearn lib where it fits the model and later creates a prediction. We check the accuracy with the `accuracy_score` method from sklearn lib. We evaluate a `confustion_matrix` via the sklearn lib to see the performance of the classification. Then we plot the data.
#### Parameters chosen
`features = [nswdemand]`
`target = class`
#### 3. Long Short Term Memory
This is a predictive model to predict `nswdemand` based on the features `features = ['date','day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']`
first we used the `MinMaxScaler` method from sklearn in order to ensure our data was normalized. We used the `Time Series Generator` method to group the data by `length=step_num` where `step_num` is `48`, because there are 48 instances for each time period of one day. We used two LSTM layers with 50 units each preceeding the dense layer with one unit. The function trains the model for 3 epochs where 1 epoch is a complete pass through the entire trained dataset. We generate predictions on the test data using the trained model. First creating a `TimeseriesGenerator` for test data and using the `predict` method to generate predictions. We plot the real data and the predictions on a line chart use matplotlib.

##### Parameters chosen
`batch_size=1000`
`epoch=3` 
`step_num=48`
`features = ['date','day', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']`
#### 4. Neural Network
##### Six hidden layers, K-Fold Cross Validation (K=5)
## Results section.
*This will include the results from the methods listed above (C). You will have figures here about your results as well.
No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.*

## Discussion section: 
*This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!*

We saw a saturation effect between `nswprice` and `nswdemand` and therefore decided that a polynomial regression would best fit the graph.
We saw values of either 1 or 0 for `UP` or `DOWN` and therefore decided to try logistic regression between `class` and `nswdemand`
## Conclusion section:
*This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts*

## Collaboration section:
*This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!
Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.*

Your final model and final results summary (this should be the last paragraph in D)
Your GitHub must be made public by the morning of the next day of the submission deadline.