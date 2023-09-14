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

### Data Exploration:
Using the pairplot method from the seaborn library, we observed a positive correlation between the features `nswdemand` and `vicdemand` as well as a saturation effect between `nswprice` and `nswdemand`. There is also a negative correlation between `transfer` and both demands (`nswdemand`, `vicdemand`)

### Preprocessing:

#### Original Movie set Data
 We are changing the date feature by omitting the dashes in the 'date' data inputs and then converting them into integers, that way we change them all into 8 digit int variables. We will also change the genre feature from an array of strings into a codefied array of int variables. It will remain an array since the movies are classified under multiple categories, but all the genres seem to repeat, so they will be easily codefied into ints. Same goes for the Production companies, they are currently array's of strings listing the production companies involved in the making of the movie, however we will just convert the string values into int values. We will drop the title column as well as the tagline column since there is no way of codefying them in a repeatable manner. For the budget column and revenue, there are about 80% of data points missing (i.e. they are listed as zero), so it is probably not possible to omitt these, so instead we probably will just ignore these columns but keep the datapoints in.

We decided to drop the tagline, title, and overview columns, and all of the nan values. We also replaced the 0's with the mean of their respective columns. We ran linear regression on the data and the error was massive, unsurprisingly. The model was trained on budget to predit revenue and likely resulted in underfitting because of the mean replacement technique we used.

#### New Electricity Data Set
&nbsp;&nbsp;&nbsp; The dataset is a timeseries dataset of electricity metrics in the Australian states of Victoria and New South Wales. The data is taken over about a 2 year period, and has 45,312 observations taken every 30 minutes. The dataset has 9 features, Class (logit that describes if demand goes up or down), Date, Day, Period (which 30 minute period the observation is), NSWprice (New South Wales price), NSWdemand (New South Wales price), VICprice (Victoria price), VICdemand (Victoria demand), and transfer (scheduled electricity transfer between both states). All of the features are normalized already  except for Day and class. The `class` feature had either values as `b'UP'` or `b'DOWN'`. Having the byte prefixed to the string made it difficult to classify and therefore was dropped via the code ```df['class'] = df['class'].str.decode('utf-8')```. We then replaced the value `UP` and `DOWN` with the numeric values `1` and `0` respectively. We then dropped null values with `df.dropna(inplace=True)`. The day label had values 1-7 prepended with the byte character. We changed the values to their respective numeric value with the map method `df['day'] = df['day'].map({b'1': 1, b'2': 2, b'3': 3, b'4': 4, b'5': 5, b'6': 6, b'7': 7})`. Most of the the data came normalized and therefore was not a required step to implement. 
 
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
#### Polynomial Regression
![](https://imageupload.io/ib/KnUO58QKIy66q1M_1694670725.png)
#### Logistic Regression
![](https://imageupload.io/ib/BsLQnHpHkzqogxO_1694671126.png)
#### Long Short Term Memory
![](https://imageupload.io/ib/pwKpt2jszS6QN7G_1694671216.png)
## Discussion section: 
*This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!*

&nbsp;&nbsp;&nbsp;For our goal of making a regression model to predict outcomes of electricity demand, the first model we ended up choosing was a Polynomial Regression model. We thought that this was a good starting point since we would build to more complex models (such as the neural net and LSTM models). For the Polynomial Regression Model, we compared the MSE's of first degree, second degree, fourth degree, and fifth degree regression models. What we found is that the fourth degree polynomial minimized our MSE for the data, and for this reason we settled on using a fourth degree polynomial regression model. In this part of the model exploration, we could have been more expansive in how many polynomial degrees we went up to. However, we judged that any more degrees would result in overfitting, so we settled on a nice and more generalized fitting of the model. We limited the features accepted by this model to 'day', 'nswprice', and 'date'. In hindsight, this was probably a little too constrictive, since we could have left in 'vicdemand' and 'vicprice', which probably could have helped contribute an associative relationship to the model to improve accuracy. A conceptual extension of the polynomial regression model for us was implementing a logistic regression model. For this model we used the 'class' parameter as the dependent variable in order to see if we could model the general ebb and flow of the swcdemand. The 'class' parameter indicates whether the demand for electricity is increasing ('1') or decreasing ('0'). We found moderate success utilizing this model with accuracy of about .6361, and a confusion matrix of 3909 True Positives, 1282 False Positives, 2016 True Negatives, and 1856 False Negatives. Our model seemed to be more accurate in predicting Positives then it was at Negatives interestingly enough. Perhaps this could be due to the fact that there are more definitive times when electricity would surge, such as when people wake up in the morning and turn on electronics, whereas times where there are drops in demand for electricity could be a bit more ambiguous with poeple going to bed at different times and having different work schedules. 

&nbsp;&nbsp;&nbsp;After exploring polynomial and logistic regression models, we turned our attention to the Neural Network model. For our NNM, we decided on an input layer with all features except for "nswprice". This is because intuitively, we felt that price would be in large part determined by demand, so it would be meaningless as a predictor since in actuality, price should be a dependent variable of demand to some degree (albiet, price elasticity of demand is where demand is affected by price, but we didn't want to muddy it too much and decided to drop the feature for the NNM entirely). Also it should be noted that in hind sight we should of probably implemented this thought process in our polynomial regression model, however there was some miscommunication in the process of making the polynomial models and strapped for time, we ended up leaving the model as is and mentioning it in the discussion section. As for the architecture of the NNM itself, we decided on 5 hidden layers, with 512, 256, 128, 64, and 32 nodes in that order. All of them use the 'relu' activation funciton except for the output layer, which used a 'linear' activation function. We tried to implement a Randomized Search CV tuning model to tune the hyperparameters of the model, but we weren't able to get around a ValueError that it kept on throwing. Theoretically, the tuning model should work in it's architecture (from what we could tell at least), but weren't able to resolve the error issues. We were planning on tuning the activation parameters between 'relu' and 'tanh', and 'sigmoid'. We were also planning on tuning the optimizers between 'adam' and 'rmsprop' and tuning kernel initializer between 'glorot_uniform' and 'normal'. We speculate that relu and adam would have performed best for our purposes, since relu would be able to piece wise together for the regression in our opinion and 'adam' would have done best at least on training efficiency since we had a very large dataset. We used k-fold cross validation on the neural net to make our training and results unbiased. Our neural net had 5 hidden layers and ran over 50 epochs. Given our large dataset, over 40k observations, using k=5 provided enough iterations to solidify our results and even showed less loss in some iterations. The overall loss became stagnant at around 0.004 loss and loss without K-Fold was 0.002, so our neural net's results showed double the accuracy without the cross validation.
&nbsp;&nbsp;&nbsp;Our next model that we decided on using was a Neural Network Model. Our parameters that we used in this model includes 'day', 'date',  We initially started off with 2 hidden layers 

&nbsp;&nbsp;&nbsp;Predictive models in any market allow companies to anticipate changes in demand and adjust their output and production for a more efficient operation. Having predicitive models can help consumers plan their usage on a product and ultimately save money. A predictive model on the electricity market can be help stabilize the electricity grid and further help balance the supply and demand. 

## Conclusion section:
*This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts*

We aimed to predict the demand of electricity using features such as price, day, period, etc. With more time and knowledge, we could have thrown more complex prediction models at the problem or tried different classification models. We also could have used unsupervised learning models for analysis instead of predicting. The importance of electricity is exponentially increasing as electricity becomes more imbedded into the lives of our technological society. Various industries necessitate electricty for their daily use and rely on the stability of these energy companies. With more advanced models and accurate results, companies would be able to minimize their risk and ensure a successful and reliable service. We hope that our attempts at predicting electricity demand helps companies outline a plan and manage their loads and costs when electricity demand is high or low.

## Collaboration section:

*This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!
Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.*

Tristan, Geoffrey, and Raymond all did their best to contribute what we could to all sections of the project, whether it be coding or writing the report. The only person who did not do ANYTHING in this entire project was Thun Zin, who did not show up for any milestones, coding, or report writing. He's also not in the repo because he never accepted the invitation. We realize the limitations of our report, but we really struggled at times to get everything done on time due to the fact that we only had 3 members to complete the entire project. 

Your final model and final results summary (this should be the last paragraph in D)
Your GitHub must be made public by the morning of the next day of the submission deadline.
