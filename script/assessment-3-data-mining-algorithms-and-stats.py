#!/usr/bin/env python
# coding: utf-8

# Assessment 3 Data Mining Algorithms
# 
# Where appropriate add two cells below the question.  One a code cell to hold any code you used, and the second a markdown cell to hold the explanation.  
# You are to submit your notebook file, which should be labelled yourname_assign3
# Note the questions are allocated different weightings, these will be scaled to 100%.

import numpy as np
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
import seaborn as sns #To visualise
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


diamonds = sns. load_dataset ("diamonds") 

diamonds.columns


# ### Diamonds metadata
# #### price 
# price in US dollars (\$326--\$18,823)<br>
# 
# #### carat 
# weight of the diamond (0.2--5.01)<br>
# 
# #### cut 
# quality of the cut (Fair, Good, Very Good, Premium, Ideal)<br>
# 
# #### color 
# diamond colour, from J (worst) to D (best)<br>
# 
# #### clarity 
# a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))<br>
# 
# #### x 
# length in mm (0--10.74)<br>
# 
# #### y 
# width in mm (0--58.9)<br>
# 
# #### z 
# depth in mm (0--31.8)<br>
# 
# #### depth 
# total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)<br>
# 
# #### table 
# width of top of diamond relative to widest point (43--95)<br>

# #### 1. In the diamonds data set identify quantitative variables that have  linear relationships
# ####  <div style="text-align: right"> (5 marks) </div><br>

# put the code you used in here


# #answer your question and give explanation in here
# 
# 

# #### 2 Create a linear regression model predicting the price of a diamond using <b><u>ONE independent variable</b></u>.<br>
# #### <div style="text-align: right"> (15 marks) </div><br>
# #### Test the different <i>quantitative</i> columns in order to identify which independent variable has the most predictive power for price.<br>
# #### In a markdown cell, provide justification as to why you chose the variable<br>
# Remember to use a reasonable split on the data to create test and train subsets 







# 




# #### 3 Create a linear regression model predicting the price of a diamond using <b><u>MULTIPLE independent variables</b></u>.
# #### <div style="text-align: right"> (15 marks) </div><br>
# 

# ## Some theory<br>
# #### Create a markdown cell under each of the following quesions, and put in your answer in there.
# 

# 4. What are dummy variables, what is the Dummy Variable Trap and how can we overcome it?<div style="text-align: right"> (5 marks) </div>

# put answer in here...

# 5. With regard to a linear regression model explain the meaning and importance of : <br>
# R^2
# ####  <div style="text-align: right"> (5 marks) </div><br>
# The coefficient (weight) associated with an independent variable
# ####  <div style="text-align: right"> (5 marks) </div><br>
# The Intercept
# ####  <div style="text-align: right"> (5 marks) </div><br>
# root mean squared error
# ####  <div style="text-align: right"> (5 marks) </div><br>
# 

# 6. The mean life of a battery is 50 hours with a standard deviation of 6 hours. The mean life of batteries follow a normal distribution.  The manufacturer advertises that they will replace all batteries that last less than 38 hours. If 100,000 batteries were produced, how many would they expect to replace?  In your answer explain your workings 
# ####  <div style="text-align: right"> (5 marks) </div><br><br>

# 7. A quality contorl process uses a grading scale to grade the quality of the batteries.  1000 batteries are produced. It is assumed that the scores are normally distributed with a mean score of 75 and a standard deviation of 15
# a)	How many batteries will have scores between 45 and 75?
# b)  If 60 is the lowest passing score, how many batteries are expected to pass the quality control check?
# In your answer, explain your workings.
# ####  <div style="text-align: right"> (10 marks) </div><br><br>
# 

# 8. The length of time the batteries are on the supermarket shelf before being sold is a mean of 12 days and a standard deviation of 3 days.  It can be assumed that the number of days on the shelf follows a normal distribution.  Answer the following questions, explai your workings for each.<br>
# a)	About what percent of the batteries remain on the shelf between 9 and 15 days?<br>
# b)	About what percent of the batteries remain on the shelf last between 12 and 15 days?<br>
# c)	About what percent of the batteries remain on the shelf last 6 days or less?<br>
# d)	About what percent of the batteries remain on the shelf last 15 or more days?<br>
# ####  <div style="text-align: right"> (10 marks) </div><br>

# 9. An online shopping store maintains the shopping history of users so that future predictions can be made about which products will appeal to which type of customer.  <br>
# The following baskets are noted. <br>
# 
#             1 ABC
#             2 ABCD
#             3 BC
#             4 ABD
#             5 BC
#             6 BCD
#             7 BD
#             8 B
#             9 A
#             10 AC
#             
# Calculate the Support and the Confidence, that a potential customer who adds A, and B to their shopping basket is likely to add product C.
# In your answer, explain your workings.
# ####  <div style="text-align: right"> (10 marks) </div><br>
# 

# 10. Which data algorithm would you choose for the following scenerios.  In your answer please explain your choice, as to why it is the most appropriate, in brief how the alogritm works, and what the expected outcomes would be. <br>
# (a) the battery company you work for is considering opening a new manufacturing plant in Europe, and has come down to the two last choices - Ireland or Poland.  You have data such as the utility costs, employment rates, mean salary for the location, and grants avaiable for the Government, such as the IDA.  Which algorithm would you use to help you choose? <br>
# (b) the software company you work for monitors users online time, access to the SaaS, number of purchases, length of time online, the number of sessions, length of session etc.  They are interested in predicting which users are likely to be retained and which are likely to churn.  What algorithm would help provide an insight to this problem? <br>
# ####  <div style="text-align: right"> (20 marks) </div><br>
# 
# 






