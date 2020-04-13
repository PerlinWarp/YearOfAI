# YearOfAI

This repo will store some of the AI work I do as progress to my new years resolution of working on more AI.
Alot of the AI work I do in my research cannot be shared and that is how I spend alot of the free time I have dedicated to AI, so this will be a small subset of all the AI I do in 2019.

I am starting off with the #100DaysOfMLCode challenge.

I am also very fond of Kaggle, [here is mine](https://www.kaggle.com/perlinwarp).

#### Day 1 - 1st of January
Tried to classify CIFAR 10 using a 4 layer model in Keras. [Link to Notebook](https://github.com/PerlinWarp/YearOfAI/blob/master/CIFAR10.ipynb)

#### Day 2 - 2nd of January
Added a 6 layer model, trained it and compared accuracy between the 2 models.

#### Day 3 - 3rd of January
Tried to figure out why my model sucks so badly.

#### Day 4 - 4rd of January
Made a new model and got 84.98% accuracy.

#### Day 5 - 5th of January
Started to work on the CIFAR100 Dataset.

#### Day 6 - 6th of Janurary
Made a model for CIFAR 100 using the 100 labels, achieving validation accuracy of 57.73% but accuracy of 0.7249.

#### Day 7 - 7th of Janurary
Playing with hyper-params and making a program for facial detection.

#### Day 8 - 8th of Janurary
Creating bounding boxes for faces from CIFAR-100 Dataset.
[Link to Notebook](https://github.com/PerlinWarp/YearOfAI/blob/master/BoundingBoxes/BoundingBoxes.ipynb)

#### Day 9 - 9th of Janurary
Used leaky relu and managed to get 60% accuracy.
[Link to Notebook](https://github.com/PerlinWarp/YearOfAI/blob/master/CIFAR100/CIFAR100-6.ipynb)

#### Day 10 - 10th of January
Looked into training my own haar cascade classifier.

#### Day 11 - 11th of January
Looking into different arcitectures and training different models for CIFAR100

#### Day 12 - 12th of January
Learning about RNNs.

#### Day 13 - 13th of January
[Kaggle Timeseries prediction with Siraj Raval.](https://www.kaggle.com/learn/time-series-with-siraj)

#### Day 14 - 14th of January
Stock price prediction.

#### Day 15 - 15th of January
Read up on ARIMA, Moving averages, Single Exponential smoothing, Holt's linear and Holts Winter methods.

#### Day 16 - 16th of January
Read Man vs Big Data

#### Day 17 - 17th of January
Started Quantopian Lectures

#### Day 18 - 18th of January
[Quantopian Tutorial 1](https://www.quantopian.com/tutorials/getting-started)

#### Day 19 - 19th of January
More quantopian tutorials and also setting up Zipline.

#### Day 20 - 20th of January
Zipline tutorials and making a Neural Net to predict crypto.

#### Day 21 - 21th of January
Data preperation for the neural net.

#### Day 22 - 22th of January
Understanding input sizes for an LSTM.

#### Day 23 - 23th of January
Made an LSTM but only got 47% accuracy.

#### Day 24 - 24th of January
Setting up a Deep learning server, a MongoDB server and trying to create a Deep Learning pipeline.

#### Day 25 - 25th of January
Making a backtester.

#### Day 26 - 26th of January
Working on super secret trading algorithms for herobots.

#### Day 27 - 27th of January
Working on the super secret trading algorithms for herobots.

#### Day 28 - 28th of January
Learned about the importance of not using shuffle to create test/train splits. Scaling and normalising data. Then implemented a preprocessing algorithm in python.

#### Day 29 - 29th of January
Learned Tensorboard and started to remake a model using different data.

#### Day 30 - 30th of January
Looked into different moving averages for modelling. Also looked into GANs.

#### Day 31 - 31th of January
[Data Lit](https://www.youtube.com/watch?v=3Pzni2yfGUQ)

#### Day 32 - 1st of February
Working on data preprocessing and gathering BTC price data. Checking for missing data.

#### Day 33 - 2nd of February
Looked info reinforcement learning.

#### Day 34 - 3rd of February
Data cleaning using Pandas.

#### Day 35 - 4rd of February
[Data Lit 2- Intro to Statistics ](https://www.youtube.com/watch?v=MdHtK7CWpCQ)

#### Day 36 - 5th of February
More data cleaning.

#### Day 37 - 6th of February
Checking for time continutity in 2 time series datasets of over 750,000 points each.

#### Day 38 - 7th of February
Fitting hyper parameters for BTC prediction model.

#### Day 39 - 8th of February
Connected to a new MongoDB server, lined up different time series in pandas while checking for missing results. Started to make a pipeline to pull data minutely and run a LTSM on it.

#### Day 40 - 9th of February
Started to set up a machine learning pipeline that will pull data, run a model and submit the predictions to a database.

#### Day 41 - 10th of February
Continued working on a pipeline.

#### Day 42 - 11th of February
Continued working on a pipeline, gathered and cleaned more data sources.

#### Day 43 - 12th of February
Started making a model with more data sources.

#### Day 44 - 13th of February
Deployed a test version of the algorithm to the server, set up authentification and cronjobs.

#### Day 45 - 14th of February
Backtesting and changing the algorithms.

#### Day 46 - 15th of February
Making a new LSTM model using more data. Looked into different ways to automate [hyperparameter picking](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/).

#### Day 47 - 16th of February
Looked at the effect of equalising the amount of data I have in each catagory on classification validation accuracy, changed hyperparameters.

#### Day 48 - 17th of February
When pulling the data from apis to run the model on, most of the values are NaNs, today was spent figuirng out why. I also remade data preprocessing algorithms.

#### Day 49 - 18th of February
Deplotying and testing a new LSTM on the BTC data.

#### Day 50 - 19th of February
Making a new model for ETH data, trying new arcitectures and seeing the effect of removing sparse data sources to give more data overall.

#### Day 51 - 20th of February
Testing out more new models.

#### Day 52 - 21th of February
Looked into decision trees and the mathematics behind neural networks.

#### Day 53 - 22th of February
Migrating to a MongoDB database to fix the problem of getting NaN values so we can make a Decision Tree.

#### Day 54 - 23th of February
Wrote a script to use a backup model for when I am creating a new model so the GPU is busy and cannot commit.

#### Day 55 - 24th of February
Started to make a decision tree.

#### Day 56 - 25th of February
Made backups of all of my Python Notebooks from my summer work. Allowing me to search through and reuse my models but also migrate to a new server.
Started Fast.ai Practical Deeplearning for coders. 

#### Day 57 - 26th of February
Learned about k fold validation and Reinforcement learning.

#### Day 58 - 27th of February
Calculated decision trees manually calculating entropy and gain.

#### Day 59 - 28th of Febuary
Looked into the mathematics behind Reinforcement learning.

#### Day 60 - 1st of March
Started making an accuracy checker for different machine learning models. 

#### Day 61 - 2nd of March 
Continued making the accuracy checker and found bugs in the API we are using. 

#### Day 62 - 3rd of March 
Finished work on the accuracy checker. 

#### Day 63 - 4rd of March 
Looked into using a SAT solver to create a suduko AI. 

#### Day 64 - 5th of March 
Made a 3 step system for testing ML models and backing up cronjobs. 

#### Day 65 - 6th of March 
Compared our different models using the accuracy checker.

#### Day 66 - 7th of March
Adding Sentry tracking to testing submissions. 

#### Day 67 - 11th of March
Adding Sentry tracking to BTC submissions and backup submissions incase of errors. 

#### Day 68 - 25th of March
After taking some time off for personal reasons, today I got back into things. Added some more models and checked my progress for herobots planning what I need to do next. 

#### Day 69 - 26th of March
Switched to a new model for ETH prediction. 
Created a ETH spot check using Logistic Regression, Random Forests, KNN, Naive Bayes and XGBoost. 
Looked at the feature importances for XGBoost.

#### Day 70 - 27th of March
Moved the new model for ETH prediction to testing. 
Added documentation. 
Set up logging into the database submissions if prediction fails. 

#### Day 71 - 28th of March
Retested old LSTM models and algorithmically created new features, creating a pipeline to test XGBoost models. 

#### Day 72 - 29th of March
Made various Logistic Regression models but most of my day was spent making my own backtester. Then creating a fuly automated pipeline that creates models for me, backtests them and then saves the results. I am very happy. 

#### Day 73 - 30th of March
Watched Statistics lectures. 
Made more XGBoost models, created and backtested many more models and added more features to my model creator. 

#### Day 74 - 31th of March
Watched lectures on Autocorrelation, learned about how variance estimates are different for autocorrelated data. Tested data for auto correletion. Reran my auto model generator and backtester on ETH. 
Derived Perceptrons from scratch. 

#### Day 75 - 1st of April
Manually made a new model that predicts with 100%. 
... And solved P = NP, well P != NP. 
Presented a demo of Sigmoids and Universal Apprximation Theory. 

#### Day 76 - 2st of April
Made an AR model and more logistic regression models with added features. 

#### Day 77 - 3rd of April
[PyTorch 60 Minute blitz](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)

#### Day 78 - 4th of April 
ExplAIn

#### Day 79 - 5th of April 
ExplAIn

#### Day 80 - 6th of April
Showing how a linear line is equlivant to a perceptron without an activation function. 

#### Day 81 - 7th of April
Using my Perceptron Demo to solve XOR.

#### Day 82 - 8th of April   
Fast AI - Meet up 7

Also watched:
https://ml4a.github.io/classes/itp-F18/06/

#### Day 85 - 11th of April
Although creating Conway's Game of Life, which practically started the field of Cellular Autonama, would have been enough for one life to be will lived. It was just one small detour on your journey of mathematical genius. Seeing this as a child lead me to where I am now and gave me a passion for alife. 
R.I.P John Conway.

On this day I made an implimentation of Conways Game of life and started CodeLife, a project in alife. 

#### Day 86 - 12th of April
Created the player and perceptron agents for codelife. 

#### Day 87 - 13th of April
![RL Codelife](media/codelife_rl.gif)
