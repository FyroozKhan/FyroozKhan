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
