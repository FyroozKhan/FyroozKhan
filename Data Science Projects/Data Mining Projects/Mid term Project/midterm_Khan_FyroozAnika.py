# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'
# First, install the rfit package using pip install rfit
# This package (rfit - regression.fit) is for use at The George Washington University Data Science program. 
# There are some useful and convenient functions we use in our python classes.
# One of them is dfapi, which connects to the api.regression.fit endpoint, to load data frames used in our classes.

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 

# world1 = rfit.dfapi('World1', 'id')
# world1.to_csv("world1.csv")
world1 = pd.read_csv("world1.csv", index_col="id") # use this instead of hitting the server if csv is on local

# world2 = rfit.dfapi('World2', 'id')
# world2.to_csv("world2.csv")
world2 = pd.read_csv("world2.csv", index_col="id") # use this instead of hitting the server if csv is on local

print("\nReady to continue.")


#%%[markdown]
# # Two Worlds 
# 
# I was searching for utopia, and came to this conclusion: If you want to do it right, do it yourself. 
# So I created two worlds. 
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
# 
# Please do whatever analysis you need, convince your audience both, one, or none of these 
# worlds is fair, or close to a utopia. 
# Use plots, maybe pivot tables, and statistical tests (optional), whatever you deem appropriate 
# and convincing, to draw your conclusions. 
# 
# There are no must-dos (except plots), should-dos, cannot-dos. The more convenicing your analysis, 
# the higher the grade. It's an art.
#

#%%

#For me, utopia means a world where there is no division, no chaos, no imbalance, no discrimination.
# It’s a world where peace and prosperity exist and where there is equity and a sense of safety and security
# for everybody. This sort of equality can exist when people grow up in similar surroundings and environments 
# and when people get similar opportunities in their lives. For example, they get similar educational and equal 
# income earning opportunities.

# Theoretically, it is possible to think of having these but practically it may not be possible to have all these
# altogether in society. Society is made up of people coming from different backgrounds and it is impossible for the
# government to create total equality among them. Hence, no society is perfect, or can it ever be and that’s because
# human beings are imperfect themselves (Cambridge English Dictionary, n.d.; Merriam-Webster, n.d.).

# Now the two datasets that we have got are world1 and world2. We can understand if these worlds are utopian or not
# based on the analysis of the given variables. We can analyze the distribution of age, gender, ethnicity, and marital
# status to identify any disparities. A utopian society would strive for equal representation across all demographics.
# Again, a high level of education for all demographics would be indicative of a more equitable society. We can compare
# the distribution of individuals across different industries in both worlds. A diverse economy with representation in
# various sectors might suggest a more resilient and prosperous society. Again, we can test the income inequality of the
# two datasets.

# To demonstrate that the two datasets world1 and world2 represent a utopia, we need to analyze and compare the
# data based on several criteria. This can include equality of income, education, employment in higher-paying
# industries, and balanced representation across gender, marital status, and ethnic groups. We are going to follow
# the following Step-by-Step Analysis:

# 1. Check for Income Equality
# 2. Check for Equal Representation in Education
# 3. Check for Balanced Marital Status
# 4. Check for Gender Balance
# 5. Check for Ethnic Diversity
# 6. Check for Employment in Higher-Paying Industries
# 7. Check for Age distribution
# 7. Combine Results to Conclude Utopia

# First, we inspect the data
print(world1.head())
print(world2.head())
print(world1.columns)
print(world2.columns)


# Second, we label the categorical variables.
# Mapping dictionaries for categorical variables
mapping_dicts = {
    'marital': {0: 'Never married', 1: 'Married', 2: 'Divorced', 3: 'Widowed'},
    'gender': {0: 'Female', 1: 'Male'},
    'ethnic': {0: 'Ethnic 0', 1: 'Ethnic 1', 2: 'Ethnic 2'},
    'industry': {
        0: 'Leisure and hospitality', 1: 'Retail', 2: 'Education', 
        3: 'Health', 4: 'Construction', 5: 'Manufacturing', 
        6: 'Professional and business', 7: 'Finance'
    }
}

# Function to apply the mappings
def apply_mappings(df, mappings):
    for column, mapping in mappings.items():
        df[column] = df[column].replace(mapping)
    return df

# Apply mappings to both datasets
world1 = apply_mappings(world1, mapping_dicts)
world2 = apply_mappings(world2, mapping_dicts)

# Display the first few rows of the transformed datasets to check the mappings
print(world1.head())
print(world2.head())



# 1. Check for Income Equality
# Calculate mean and median income for both datasets
mean_income_world1 = world1['income'].mean()
median_income_world1 = world1['income'].median()

mean_income_world2 = world2['income'].mean()
median_income_world2 = world2['income'].median()

print(f"World1 - Mean Income: {mean_income_world1}, Median Income: {median_income_world1}")
print(f"World2 - Mean Income: {mean_income_world2}, Median Income: {median_income_world2}")

# Here, the means and medians are quite close in both datasets, suggesting that the overall income
# levels are similar.


# Visualize Income Distribution

# Histogram
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.hist(world1['income'], bins=30, alpha=0.5, label='World1')
plt.hist(world2['income'], bins=30, alpha=0.5, label='World2')
plt.title('Income Distribution - Histogram')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.legend()

# Box Plot
plt.subplot(1, 3, 2)
plt.boxplot([world1['income'], world2['income']], labels=['World1', 'World2'])
plt.title('Income Distribution - Box Plot')
plt.ylabel('Income')

# Density Plot
plt.subplot(1, 3, 3)
sns.kdeplot(world1['income'], label='World1', shade=True, alpha=0.5)
sns.kdeplot(world2['income'], label='World2', shade=True, alpha=0.5)
plt.title('Income Distribution - Density Plot')
plt.xlabel('Income')
plt.legend()

plt.tight_layout()
plt.show()

# The visualizations suggest that the income distributions are more right skewed and have 
# extreme values towards the upper limit. This indicates inequality in both the worlds.
# Now let's test if the difference in means is significant or not.

# As the data is not normally distributed, we perform Mann-Whitney U Test to  to compare the distributions of income:
# Extract the income columns
income1 = world1['income']
income2 = world2['income']

# Perform the Mann-Whitney U Test
from scipy.stats import mannwhitneyu

u_statistic, p_value = mannwhitneyu(income1, income2)
print(f"Mann-Whitney U Statistic: {u_statistic}, P-value: {p_value}")

# The results suggest that Given the p-value of 0.7649, we can conclude that the income distributions
# in world1 and world2 are not significantly different. This suggests that neither world can be definitively
# classified as more utopian than the other based solely on income distribution, as they have similar
# characteristics in terms of income levels.


# 2. Check for Income Distribution by Gender

print("World 1 Gender Counts:")
print(world1['gender'].value_counts())

print("\nWorld 2 Gender Counts:")
print(world2['gender'].value_counts())

# Shape the data for easier plotting
world1_melted = pd.melt(world1, id_vars=['gender'], value_vars=['income'], var_name='variable', value_name='value')
world2_melted = pd.melt(world2, id_vars=['gender'], value_vars=['income'], var_name='variable', value_name='value')

# Create boxplot for World 1
plt.figure(figsize=(12, 6))
sns.boxplot(x='gender', y='value', data=world1_melted, palette="Blues", showfliers=False)
plt.xticks([0, 1], ['Female', 'Male'])
plt.title('Income Distribution by Gender in World 1')
plt.ylabel('Income')
plt.grid()
plt.show()

# Create boxplot for World 2
plt.figure(figsize=(12, 6))
sns.boxplot(x='gender', y='value', data=world2_melted, palette="Oranges", showfliers=False)
plt.xticks([0, 1], ['Female', 'Male'])
plt.title('Income Distribution by Gender in World 2')
plt.ylabel('Income')
plt.grid()
plt.show()

# The visualizations show that the income of the female group is higher than the male group for World1.
# So the world1 is not utopian in this aspect. Whereas, the income is fairly equally distributed for 
# world2. So, it seems to be utopian. Regardless of this, we will check other variables:

# 3. Check for Income Distribution by Education level

def categorize_education(years):
    if years == 0:
        return 'no formal education'
    elif 1 <= years <= 8:
        return 'elementary education'
    elif 9 <= years <= 12:
        return 'high school education'
    elif 13 <= years <= 15:
        return 'college level education'
    elif years == 16:
        return "bachelor's degree"
    elif 17 <= years <= 18:
        return "master's degree"
    elif years > 18:
        return "doctorate degree"
    else:
        return 'unknown'

# Apply the categorization function to the education column in both datasets
world1['education_category'] = world1['education'].apply(categorize_education)
world2['education_category'] = world2['education'].apply(categorize_education)

# Boxplot for Income Distribution by Categorized Education in World 1
plt.figure(figsize=(14, 8))
sns.boxplot(x='education_category', y='income', data=world1, palette="Blues", showfliers=False)
plt.title('Income Distribution by Categorized Education in World 1')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Boxplot for Income Distribution by Categorized Education in World 2
plt.figure(figsize=(14, 8))
sns.boxplot(x='education_category', y='income', data=world2, palette="Oranges", showfliers=False)
plt.title('Income Distribution by Categorized Education in World 2')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Histogram for Income Distribution by Categorized Education in World 1
g = sns.FacetGrid(world1, col="education_category", col_wrap=4, height=4, aspect=1.5, palette="Blues")
g.map(sns.histplot, "income", bins=30)
g.fig.suptitle('Income Distribution by Categorized Education in World 1', y=1.03)
g.set_axis_labels('Income', 'Count')
plt.show()

# Histogram for Income Distribution by Categorized Education in World 2
g = sns.FacetGrid(world2, col="education_category", col_wrap=4, height=4, aspect=1.5, palette="Oranges")
g.map(sns.histplot, "income", bins=30)
g.fig.suptitle('Income Distribution by Categorized Education in World 2', y=1.03)
g.set_axis_labels('Income', 'Count')
plt.show()

# We do see difference in the median income of the different education levels between the two worlds,
# particularly for the education levels with no formal educationa and elementary education. Now, let's
# test if this differene is significant or not.
# Performing One-Way ANOVA to compare each education category separately

from scipy.stats import f_oneway

# Combine the datasets and create an indicator variable for the world
world1['world'] = 'world1'
world2['world'] = 'world2'
combined = pd.concat([world1, world2])

# List of unique education categories
education_categories = combined['education_category'].unique()

# Dictionary to store ANOVA results
anova_results = {}

# Perform One-Way ANOVA for each education category
for category in education_categories:
    income_world1 = combined[(combined['education_category'] == category) & (combined['world'] == 'world1')]['income']
    income_world2 = combined[(combined['education_category'] == category) & (combined['world'] == 'world2')]['income']
    
    # Perform ANOVA
    f_statistic, p_value = f_oneway(income_world1, income_world2)
    
    # Store the results
    anova_results[category] = {'F-Statistic': f_statistic, 'P-Value': p_value}

# Display the ANOVA results
for category, result in anova_results.items():
    print(f"Education Category: {category}")
    print(f"  F-Statistic: {result['F-Statistic']}")
    print(f"  P-Value: {result['P-Value']}\n")


# The box plots show the the income distribution is same over the given industries.
# But the income distribution is not similar across all industries which bring us the doubt if the worlds
# are utopian or not.

# 2. Check for Equal Representation in Education
# Categorizing education levels

def categorize_education(years):
    if years == 0:
        return 'no formal education'
    elif 1 <= years <= 8:
        return 'elementary education'
    elif 9 <= years <= 12:
        return 'high school education'
    elif 13 <= years <= 15:
        return 'college level education'
    elif years == 16:
        return "bachelor's degree"
    elif 17 <= years <= 18:
        return "master's degree"
    elif years > 18:
        return "doctorate degree"
    else:
        return 'unknown'

# Apply the categorization function to the education column in both datasets
world1['education_category'] = world1['education'].apply(categorize_education)
world2['education_category'] = world2['education'].apply(categorize_education)

# Ensure the education categories are ordered correctly
education_order = [
    'no formal education', 
    'elementary education', 
    'high school education', 
    'college level education', 
    "bachelor's degree", 
    "master's degree", 
    'doctorate degree'
]

# Plot boxplotsfor World 1
plt.figure(figsize=(12, 6))
sns.boxplot(x='education_category', y='income', data=world1, order=education_order, palette="Blues")
plt.title('Income Distribution by Categorized Education in World 1')
plt.xlabel('Categorized Education')
plt.ylabel('Income')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Plot boxplot for World 2
plt.figure(figsize=(12, 6))
sns.boxplot(x='education_category', y='income', data=world2, order=education_order, palette="Oranges")
plt.title('Income Distribution by Categorized Education in World 2')
plt.xlabel('Categorized Education')
plt.ylabel('Income')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# These plots show difference for some elementary education field.

# Boxplots
plt.figure(figsize=(14, 7))
sns.boxplot(data=world1, x='education_category', y='income', palette='Blues', showfliers=False)
plt.xticks(rotation=45)
plt.title('Income Distribution by Education Category in World 1')
plt.xlabel('Education Category')
plt.ylabel('Income')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=world2, x='education_category', y='income', palette='Oranges', showfliers=False)
plt.xticks(rotation=45)
plt.title('Income Distribution by Education Category in World 2')
plt.xlabel('Education Category')
plt.ylabel('Income')
plt.grid(True)
plt.show()


# Now let's test these visualizations based on ethnic and gender variables:

# Set plot style for better aesthetics
sns.set(style="whitegrid")

# Boxplot for Income Distribution by Education Level, Gender, and Ethnicity in World 1
plt.figure(figsize=(14, 8))
sns.boxplot(x='education_category', y='income', hue='gender', data=world1, palette="Blues", showfliers=False)
plt.title('Income Distribution by Education Level and Gender in World 1')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.grid()
plt.legend(title='Gender')
plt.show()

# Boxplot for Income Distribution by Education Level, Gender, and Ethnicity in World 2
plt.figure(figsize=(14, 8))
sns.boxplot(x='education_category', y='income', hue='gender', data=world2, palette="Oranges", showfliers=False)
plt.title('Income Distribution by Education Level and Gender in World 2')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.grid()
plt.legend(title='Gender')
plt.show()

# Boxplot for Income Distribution by Education Level and Ethnicity in World 1
plt.figure(figsize=(14, 8))
sns.boxplot(x='education_category', y='income', hue='ethnic', data=world1, palette="Blues", showfliers=False)
plt.title('Income Distribution by Education Level and Ethnicity in World 1')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.grid()
plt.legend(title='Ethnicity')
plt.show()

# Boxplot for Income Distribution by Education Level and Ethnicity in World 2
plt.figure(figsize=(14, 8))
sns.boxplot(x='education_category', y='income', hue='ethnic', data=world2, palette="Oranges", showfliers=False)
plt.title('Income Distribution by Education Level and Ethnicity in World 2')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.grid()
plt.legend(title='Ethnicity')
plt.show()

# All these visualizations are self-explanatory in the sense that they clearly depict that there is difference 
# in the education level and income of different ethinicity and genders among and between both the worlds.

# So altogther, it can be said with certainty that none of the worlds are utopian. They both have difference in
# either income level, or education level at gender and ethnic levels.