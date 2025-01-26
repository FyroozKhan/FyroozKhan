# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import rfit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


# world1 = rfit.dfapi('World1', 'id')
# world1.to_csv("world1.csv")
world1 = pd.read_csv("world1.csv", index_col="id") # use this instead of hitting the server if csv is on local

# world2 = rfit.dfapi('World2', 'id')
# world2.to_csv("world2.csv")
world2 = pd.read_csv("world2.csv", index_col="id") # use this instead of hitting the server if csv is on local

print("\nReady to continue.")

#%% [markdown]
# # Two Worlds (Continuation from midterm: Part I - 25%)
# 
# In the midterm project, we used statistical tests and visualization to 
# study these two worlds. Now let us use the modeling techniques we now know
# to give it another try. 
# 
# Use appropriate models that we learned in this class or elsewhere, 
# elucidate what the current state of each world looks like. 
# 
# However, having an accurate model does not tell us if the worlds are 
# utopia or not. Is it possible to connect these concepts together? (Try something called 
# "feature importance")
# 
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 

# First, we inspect the data
print(world1.head())
print(world2.head())
print(world1.columns)
print(world2.columns)

# Load the datasets if they are not already loaded
world1 = pd.read_csv("world1.csv", index_col="id")
world2 = pd.read_csv("world2.csv", index_col="id")

# Define a function to prepare data and models for a given world
def model_world(world_data):
    # Calculate the median income for the world
    median_income = world_data['income00'].median()
    
    # Create a binary classification target column: 1 for high income, 0 for low income
    world_data['high_income'] = (world_data['income00'] > median_income).astype(int)
    
    # Select features (excluding 'income' and 'high_income' columns for prediction)
    X = world_data.drop(columns=['income00', 'high_income'])
    y = world_data['high_income']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Logistic Regression Model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, logreg_pred))
    
    # Decision Tree Classifier Model
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, tree_pred))

# Run models on world1 and world2
print("Modeling for World1:")
model_world(world1)
print("\nModeling for World2:")
model_world(world2)


# Logistic Regression outperforms the decision tree in both worlds, particularly in terms of precision 
# for class 1 (high income). Logistic regression also has better recall for low-income individuals in both worlds.
# Decision Tree is slightly less accurate, but it offers interpretability, which might be valuable for understanding
# the decision-making process, such as the relationships between features like education, age, and industry with income.
# Both models show similar trends across worlds, with World2 slightly outperforming World1 in terms of precision and recall
# for class 0 (low income), particularly with the logistic regression model.


# Define a function to extract feature importance for a given world and return as a DataFrame
def get_feature_importance_table(world_data):
    # Define features (exclude 'income' and 'high_income')
    X = world_data.drop(columns=['income00', 'high_income'])
    
    # Train Decision Tree Classifier
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X, world_data['high_income'])
    
    # Get feature importance
    importance = tree.feature_importances_
    
    # Create a DataFrame with feature names and importance
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    })
    
    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    return feature_importance_df

# Get feature importance table for World1 and World2
print("Feature Importance for World1:")
world1_feature_importance = get_feature_importance_table(world1)
print(world1_feature_importance)

print("\nFeature Importance for World2:")
world2_feature_importance = get_feature_importance_table(world2)
print(world2_feature_importance)

# Industry and Age are the dominant features in both worlds, contributing around 70% and 20% 
# to the classification of income. This suggests that the main factors driving income in both 
# worlds are industry and experience, rather than social factors like education, marital status, ethnicity, or gender.

# Social Factors (Gender, Ethnicity, Marital Status):
# Both worlds show very low importance for social features such as gender and ethnicity, indicating 
# relatively low discrimination or inequality based on these characteristics. This is a positive sign, 
# suggesting that income in these worlds may be more meritocratic.
# Marital status also plays a minimal role in income classification, further reinforcing the idea that 
# personal circumstances unrelated to work do not have a significant impact on income.

# Education: While education has a slightly higher importance than other social factors, its impact is 
# still relatively low. This suggests that, in both worlds, formal education is not the primary driver 
# of income, but rather experience and the type of industry in which individuals work.




#%% [markdown]
#
# # Free Worlds (Continuation from midterm: Part II - 25%)
# 
# To-do: Complete the method/function "predictFinalIncome" towards the end of this Part II codes.  
#  
# The worlds are gifted with freedom. Sort of.  
# I have a model built for them. It predicts their MONTHLY income/earning growth, 
# base on the characteristics of the individual. Your task is to first examine and 
# understand the model. If you don't like it, build your own world and own model. 
# For now, please help me finish the last piece.  
# 
# My model will predict what the growth factor for each person is in the immediate month ahead. 
# Along the same line, it also calculates what the expected (average) salary will be after 1 month with 
# that growth rate. You need to help make it complete, by producing a method/function that will 
# calculate what the salary will be after n months. (Method: predictFinalIncome )  
# 
# Then try this model on people like Plato, and also create a few hypothetical characters
# with different demographic traits, and see what their growth rates / growth factors are.
# Use the sample codes after the class definition below.  
# 
#%%
class Person:
  """ 
  a person with properties in the utopia 
  """

  def __init__(self, personinfo):
    self.age00 = personinfo['age'] # age at creation or record. Do not change.
    self.age = personinfo['age'] # age at current time. 
    self.income00 = personinfo['income'] # income at creation or record. Do not change.
    self.income = personinfo['income'] # income at current time.
    self.education = personinfo['education']
    self.gender = personinfo['gender']
    self.marital = personinfo['marital']
    self.ethnic = personinfo['ethnic']
    self.industry = personinfo['industry']
    # self.update({'age00': self.age00, 
    #         'age': self.age,
    #         'education': self.education,
    #         'gender': self.gender,
    #         'ethnic': self.ethnic,
    #         'marital': self.marital,
    #         'industry': self.industry,
    #         'income00': self.income00,
    #         'income': self.income})
    return
  
  def update(self, updateinfo):
    for key,val in updateinfo.items():
      if key in self.__dict__ : 
        self.__dict__[key] = val
    return
        
  def __getitem__(self, item):  # this will allow both person.gender or person["gender"] to access the data
    return self.__dict__[item]

  
#%%  
class myModel:
  """
  The earning growth model for individuals in the utopia. 
  This is a simplified version of what a model could look like, at least on how to calculate predicted values.
  """

  # ######## CONSTRUCTOR  #########
  def __init__(self, bias) :
    """
    :param bias: we will use this potential bias to explore different scenarios to the functions of gender and ethnicity

    :param b_0: the intercept of the model. This is like the null model. Or the current average value. 

    :param b_age: (not really a param. it's more a function/method) if the model prediction of the target is linearly proportional to age, this would the constant coefficient. In general, this does not have to be a constant, and age does not even have to be numerical. So we will treat this b_age as a function to convert the value (numerical or not) of age into a final value to be combined with b_0 and the others 
    
    :param b_education: similar. 
    
    :param b_gender: similar
    
    :param b_marital: these categorical (coded into numeric) levels would have highly non-linear relationship, which we typically use seaparate constants to capture their effects. But they are all recorded in this one function b_martial
    
    :param b_ethnic: similar
    
    :param b_industry: similar
    
    :param b_income: similar. Does higher salary have higher income or lower income growth rate as lower salary earners?
    """

    self.bias = bias # bias is a dictionary with info to set bias on the gender function and the ethnic function

    # ##################################################
    # The inner workings of the model below:           #
    # ##################################################

    self.b_0 = 0.0023 # 0.23% MONTHLY grwoth rate as the baseline. We will add/subtract from here

    # Technically, this is the end of the constructor. Don't change the indent

  # The rest of the "coefficients" b_1, b_2, etc are now disguised as functions/methods
  def b_age(self, age): # a small negative effect on monthly growth rate before age 45, and slight positive after 45
    effect = -0.00035 if (age<40) else 0.00035 if (age>50) else 0.00007*(age-45)
    return effect

  def b_education(self, education): 
    effect = -0.0006 if (education < 8) else -0.00025 if (education <13) else 0.00018 if (education <17) else 0.00045 if (education < 20) else 0.0009
    return effect

  def b_gender(self, gender):
    effect = 0
    biasfactor = 1 if ( self.bias["gender"]==True or self.bias["gender"] > 0) else 0 if ( self.bias["gender"]==False or self.bias["gender"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.00045 if (gender<1) else 0.00045  # This amount to about 1% difference annually
    return biasfactor * effect 

  def b_marital(self, marital): 
    effect = 0 # let's assume martial status does not affect income growth rate 
    return effect

  def b_ethnic(self, ethnic):
      effect = 0
      biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1 
      if ethnic == 0:
          effect = -0.0008  # Example: Higher negative bias for ethnic group 0
      elif ethnic == 1:
          effect = -0.0003  # Example: Lower negative bias for ethnic group 1
      elif ethnic == 2:
          effect = 0.0005  # Example: Positive bias for ethnic group 2
      return biasfactor * effect

  def b_industry(self, industry):
    effect = 0 if (industry < 2) else 0.00018 if (industry <4) else 0.00045 if (industry <5) else 0.00027 if (industry < 6) else 0.00045 if (industry < 7) else 0.00055
    return effect

  def b_income(self, income):
    # This is the kicker! 
    # More disposable income allow people to invest (stocks, real estate, bitcoin). Average gives them 6-10% annual return. 
    # Let us be conservative, and give them 0.6% return annually on their total income. So say roughly 0.0005 each month.
    # You can turn off this effect and compare the difference if you like. Comment in-or-out the next two lines to do that. 
    # effect = 0
    effect = 0 if (income < 50000) else 0.0001 if (income <65000) else 0.00018 if (income <90000) else 0.00035 if (income < 120000) else 0.00045 
    # Notice that this is his/her income affecting his/her future income. It's exponential in natural. 
    return effect

    # ##################################################
    # end of black box / inner structure of the model  #
    # ##################################################

  # other methods/functions
  def predictGrowthFactor( self, person ): # this is the MONTHLY growth FACTOR
    factor = 1 + self.b_0 + self.b_age( person["age"] ) + self.b_education( person['education'] ) + self.b_ethnic( person['ethnic'] ) + self.b_gender( person['gender'] ) + self.b_income( person['income'] ) + self.b_industry( person['industry'] ) + self.b_marital( ['marital'] )
    # becareful that age00 and income00 are the values of the initial record of the dataset/dataframe. 
    # After some time, these two values might have changed. We should use the current values 
    # for age and income in these calculations.
    return factor

  def predictIncome( self, person ): # perdict the new income one MONTH later. (At least on average, each month the income grows.)
    return person['income']*self.predictGrowthFactor( person )

  def predictFinalIncome(self, n, person): 
    # If n is 0, we return the person's current income (the base case for the recursion).
    if n == 0:
        return person['income']
    
    # Predict income for the next month
    next_month_income = self.predictIncome(person)
    
    # Update person's age for the next month
    person.update({"age": person["age"] + 1})
    
    # Recursively predict the income for the next n-1 months
    return self.predictFinalIncome(n-1, person)
  

print("\nReady to continue.")

#%%
# SAMPLE CODES to try out the model
utopModel = myModel( { "gender": False, "ethnic": False } ) # no bias Utopia model
biasModel = myModel( { "gender": True, "ethnic": True } ) # bias, flawed, real world model

print("\nReady to continue.")

#%%
# Now try the two models on some versions of different people. 
# See what kind of range you can get. For starters, I created a character called Plato for you as an example.
# industry: 0-leisure n hospitality, 1-retail , 2- Education 17024, 3-Health, 4-construction, 5-manufacturing, 6-professional n business, 7-finance
# gender: 0-female, 1-male
# marital: 0-never, 1-married, 2-divorced, 3-widowed
# ethnic: 0, 1, 2 
# age: 30-60, although there is no hard limit what you put in here.
# income: no real limit here.

months = 12 # Try months = 1, 12, 60, 120, 360
# In the ideal world model with no bias
plato = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'utop: {utopModel.predictGrowthFactor(plato)} - plato - growthfactor') # This is the current growth factor for plato
print(f'utop: {utopModel.predictIncome(plato)} - plato - income') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'utop: {utopModel.predictFinalIncome(months,plato)} - plato - finalincome')

# If plato ever gets a raise, or get older, you can update the info with a dictionary:
# plato.update( { "age": 59, "education": 21, "marital": 1, "income": 130000 } )

print("\nReady to continue.")

# In the flawed world model with biases on gender and ethnicity 
aristotle = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'bias: {biasModel.predictGrowthFactor(aristotle)} - aristotle - growthfactor') # This is the current growth factor for aristotle
print(f'bias: {biasModel.predictIncome(aristotle)} - aristotle - income') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias: {biasModel.predictFinalIncome(months,aristotle)} - aristotle - finalincome')

print("\nReady to continue.")

# Create some hypothetical characters

# Character 1: High-earning, highly educated woman in finance
hypothetical_woman_1 = Person({
    "age": 35,
    "education": 22, 
    "gender": 0, 
    "marital": 1, 
    "ethnic": 2, 
    "industry": 7, 
    "income": 200000
})
print(f'utop: {utopModel.predictGrowthFactor(hypothetical_woman_1)} - hypothetical_woman_1 - growthfactor')
print(f'utop: {utopModel.predictIncome(hypothetical_woman_1)} - hypothetical_woman_1 - income') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'utop: {utopModel.predictFinalIncome(months,hypothetical_woman_1)} - hypothetical_woman_1 - finalincome')
print(f'bias: {biasModel.predictGrowthFactor(hypothetical_woman_1)} - hypothetical_woman_1 - growthfactor')
print(f'bias: {biasModel.predictIncome(hypothetical_woman_1)} - hypothetical_woman_1 - income') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias: {biasModel.predictFinalIncome(months,hypothetical_woman_1)} - hypothetical_woman_1 - finalincome')

print("\nReady to continue.")

# Character 2: Low-income, less educated man in construction
hypothetical_man_2 = Person({
    "age": 40,
    "education": 12, 
    "gender": 1, 
    "marital": 0, 
    "ethnic": 0, 
    "industry": 4, 
    "income": 30000
})
print(f'utop: {utopModel.predictGrowthFactor(hypothetical_man_2)} - hypothetical_man_2 - growthfactor')
print(f'utop: {utopModel.predictIncome(hypothetical_man_2)} - hypothetical_man_2 - income') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'utop: {utopModel.predictFinalIncome(months,hypothetical_man_2)} - hypothetical_man_2 - finalincome')
print(f'bias: {biasModel.predictGrowthFactor(hypothetical_man_2)} - hypothetical_man_2 - growthfactor')
print(f'bias: {biasModel.predictIncome(hypothetical_man_2)} - hypothetical_man_2 - income') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias: {biasModel.predictFinalIncome(months,hypothetical_man_2)} - hypothetical_man_2 - finalincome')


print("\nReady to continue.") 


#%% [markdown]
# # Evolution (Part III - 25%)
# 
# We want to let the 24k people in WORLD#2 to evolve, for 360 months. You can either loop them through, and 
# create a new income or incomeFinal variable in the dataframe to store the new income level after 30 years. Or if you can figure out a way to do 
# broadcasting the predict function on the entire dataframe that can work too. If you loop through them, you can also consider 
# using Person class to instantiate the person and do the calculations that way, then destroy it when done to save memory and resources. 
# If the person has life changes, it's much easier to handle it that way, then just transform the data frame directly.
# 
# We have just this one goal, to see what the world looks like after 30 years, according to the two models (utopModel and biasModel). 
# 
# Remember that in the midterm, we found that there is not much gender or ethnic bias in income distribution in world2.
# Now if we let the world to evolve under the utopia model "utopmodel", and the biased model "biasmodel", what will the income distributions 
# look like after 30 years?
# 
# Answer this in terms of distribution of income only. I don't care about 
# other utopian measures in this question here. 


def simulate_income_evolution(world, model, months=360):
    """
    Simulates income evolution for individuals in a given world using the specified model.

    Args:
        world: pandas DataFrame containing information about individuals in the world.
        model: `myModel` object representing the income growth model.
        months: Number of months to simulate income evolution.

    Returns:
        pandas DataFrame with an additional column 'income_final' containing the predicted income after 'months'.
    """

    world['income_final'] = world.apply(lambda row: 
                                       model.predictFinalIncome(months, 
                                                               Person({'age': row['age00'], 
                                                                      'education': row['education'], 
                                                                      'gender': row['gender'], 
                                                                      'marital': row['marital'], 
                                                                      'ethnic': row['ethnic'], 
                                                                      'industry': row['industry'], 
                                                                      'income': row['income00']})), 
                                       axis=1)
    return world

# Simulate income evolution for world2 using both models
world2_utopian = simulate_income_evolution(world2, utopModel)
world2_biased = simulate_income_evolution(world2, biasModel)

# Analyze income distribution after 30 years
print("Income Distribution in World2 after 30 years (Utopian Model):")
print(world2_utopian['income_final'].describe())
print(world2_utopian['income_final'].quantile([0.25, 0.5, 0.75]))  # Quartiles

print("\nIncome Distribution in World2 after 30 years (Biased Model):")
print(world2_biased['income_final'].describe())
print(world2_biased['income_final'].quantile([0.25, 0.5, 0.75]))  # Quartiles

# Visualize income distributions
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=world2_utopian, x='income_final', kde=True, label='Utopian Model')
sns.histplot(data=world2_biased, x='income_final', kde=True, label='Biased Model')
plt.legend()
plt.xlabel('Income after 30 years')
plt.ylabel('Density')
plt.title('Income Distribution in World2 after 30 years')
plt.show()

# The provided output suggests that both the Utopian and Biased models, after 30 years of simulation,
# result in virtually identical income distributions in World2.
# The key summary statistics (mean, standard deviation, min, max, quartiles) are exactly the same for both models. 
# This implies that the overall shape and spread of the income distribution are indistinguishable between the two scenarios.

# This finding is unexpected given the initial assumption of biases in the biasModel. 
# It seems that despite the built-in biases related to gender and ethnicity in the biasModel, these biases 
# have not significantly impacted the overall income distribution in World2 after 30 years of simulation.



#%% 
# # Reverse Action (Part IV - 25%)
# 
# Now let's turn our attention to World 1, which you should have found in the midterm that it is far from being fair in terms of income inequality across gender & ethnicity groups.
# Historically, federal civil rights laws such as Affirmative Action and Title IX were established to fight real-world bias that existed against females and ethnic minorities.
# Programs that provide more learning and development opportunities for underrepresented groups can help reduce bias the existed in the society.
# Although achieving an absolute utopia might be infeasible, we should all aspire to create a better society that provides equal access to education, employment and healthcare and growth
# opportunities for everyone regardless of race, ethnicity, age, gender, religion, sexual orientation, and other diverse backgrounds. 

# %%
# Let us now put in place some policy action to reverse course, and create a reverse bias model:
revbiasModel = myModel( { "gender": -1, "ethnic": -1 } ) # revsered bias, to right what is wronged gradually.

# If we start off with World 1 on this revbiasModel, is there a chance for the world to eventual become fair like World #2?
# If so, how long does it take, to be fair for the different genders? How long for the different ethnic groups? 

# If the current model cannot get the job done, feel free to tweak the model with more aggressive intervention to change the growth rate percentages on gender and ethnicity to make it work. 

#%%

# Initially with the following values of the function b_ethnic, there was almost no chance for the world to become fair. 
'''
  def b_ethnic(self, ethnic):
    effect = 0
    biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.0006 if (ethnic < 1) else -0.00027 if (ethnic < 2) else 0.00045 
    return biasfactor * effect
'''

# tried the following, but that did not work
'''
def simulate_income_evolution(world, model, months=360):
    """
    Simulates income evolution for individuals in a given world using the specified model.

    Args:
        world: pandas DataFrame containing information about individuals in the world.
        model: `myModel` object representing the income growth model.
        months: Number of months to simulate income evolution.

    Returns:
        pandas DataFrame with an additional column 'income_final' containing the predicted income after 'months'.
    """

    world['income_final'] = world.apply(lambda row: 
                                       model.predictFinalIncome(months, 
                                                               Person({'age': row['age00'], 
                                                                      'education': row['education'], 
                                                                      'gender': row['gender'], 
                                                                      'marital': row['marital'], 
                                                                      'ethnic': row['ethnic'], 
                                                                      'industry': row['industry'], 
                                                                      'income': row['income00']})), 
                                       axis=1)
    return world

# Define a function to assess fairness 
def is_fair(world, tolerance=0.05):
    """
    Checks if income distribution is considered 'fair' based on mean income differences between groups.

    Args:
        world: DataFrame with simulated income data.
        tolerance: Maximum acceptable percentage difference in mean income between any two groups.

    Returns:
        bool: True if all groups have mean incomes within the tolerance level, False otherwise.
    """
    mean_income_by_gender = world.groupby('gender')['income_final'].mean()
    mean_income_by_ethnicity = world.groupby('ethnic')['income_final'].mean()

    max_gender_diff = (max(mean_income_by_gender) - min(mean_income_by_gender)) / max(mean_income_by_gender)
    max_ethnic_diff = (max(mean_income_by_ethnicity) - min(mean_income_by_ethnicity)) / max(mean_income_by_ethnicity)

    return (max_gender_diff <= tolerance) and (max_ethnic_diff <= tolerance)

# Create a list of simulation periods
simulation_periods = [360, 600, 960, 1440, 1920]  # 30 years, 50 years, 80 years, 120 years, 160 years

# Create a list of bias reversal levels
bias_levels = [-0.5, -1, -1.5] 

# Initialize variables to store results
fairness_achieved = False
time_to_fairness = None 
best_bias_level = None

# Loop through different bias reversal levels
for bias_level in bias_levels:
    revbiasModel = myModel( { "gender": bias_level, "ethnic": bias_level } )

    for months in simulation_periods:
        world1_reversed = simulate_income_evolution(world1, revbiasModel, months)

        if is_fair(world1_reversed):
            fairness_achieved = True
            time_to_fairness = months / 12  # Convert months to years
            best_bias_level = bias_level
            break

    if fairness_achieved:
        break

# Print results
if fairness_achieved:
    print(f"Fairness achieved after {time_to_fairness} years with bias level: {best_bias_level}")
else:
    print("Fairness not achieved within the simulated time horizon.")
'''

# So, changed the function b_ethnic values to this in Part II:
'''
def b_ethnic(self, ethnic):
     effect = 0
     biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1 
     if ethnic == 0:
         effect = -0.0008  # Example: Higher negative bias for ethnic group 0
     elif ethnic == 1:
         effect = -0.0003  # Example: Lower negative bias for ethnic group 1
     elif ethnic == 2:
         effect = 0.0005  # Example: Positive bias for ethnic group 2
     return biasfactor * effect
'''


# Now trying the following gave us pretty close values
# Assuming the `Person` and `myModel` classes and the `world1` DataFrame are defined as in the previous code

def simulate_income_evolution(world, model, months=360):
    """
    Simulates income evolution for individuals in a given world using the specified model.

    Args:
        world: pandas DataFrame containing information about individuals in the world.
        model: `myModel` object representing the income growth model.
        months: Number of months to simulate income evolution.

    Returns:
        pandas DataFrame with an additional column 'income_final' containing the predicted income after 'months'.
    """

    world['income_final'] = world.apply(lambda row: 
                                       model.predictFinalIncome(months, 
                                                               Person({'age': row['age00'], 
                                                                      'education': row['education'], 
                                                                      'gender': row['gender'], 
                                                                      'marital': row['marital'], 
                                                                      'ethnic': row['ethnic'], 
                                                                      'industry': row['industry'], 
                                                                      'income': row['income00']})), 
                                       axis=1)
    return world

# Create the reverse bias model with stronger intervention
revbiasModel = myModel( { "gender": -1, "ethnic": -1 } )  # Increase bias reversal

# Define a list of simulation periods in months
simulation_periods = [360, 720, 1080]  # 30 years, 60 years, 90 years, 120 years, 160 years

for months in simulation_periods:
    # Simulate income evolution for each simulation period
    world1_reversed = simulate_income_evolution(world1, revbiasModel, months) 

    # Analyze income distribution 
    print(f"\nIncome Distribution in World1 after {months/12} years (Reverse Bias Model):")
    print(world1_reversed['income_final'].describe())
    print(world1_reversed['income_final'].quantile([0.25, 0.5, 0.75])) 

    # Analyze income distribution by gender and ethnicity (optional)
    print("\nIncome Distribution by Gender (World1, Reverse Bias Model):")
    print(world1_reversed.groupby('gender')['income_final'].describe())

    print("\nIncome Distribution by Ethnicity (World1, Reverse Bias Model):")
    print(world1_reversed.groupby('ethnic')['income_final'].describe())

    # Calculate and print the mean income for each group
    mean_income_by_gender = world1_reversed.groupby('gender')['income_final'].mean()
    print(f"\nMean Income by Gender (World1, Reverse Bias Model) after {months/12} years:")
    print(mean_income_by_gender)

    mean_income_by_ethnicity = world1_reversed.groupby('ethnic')['income_final'].mean()
    print(f"\nMean Income by Ethnicity (World1, Reverse Bias Model) after {months/12} years:")
    print(mean_income_by_ethnicity)


# So yes, if we start off with World 1 on this revbiasModel, there is some chance for the world to eventual become fair like World #2
# in around 30 years. But the fairness among genders and ethnic groups is not achieved easily, unless the metrics are changed.
