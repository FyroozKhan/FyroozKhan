#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm



#%%
df = pd.read_csv("data_science_salaries.csv")
df.head()
df.tail()
df.info()
df.describe()
df.describe(include=[object])
df.shape
df.dtypes
df.duplicated().sum()
#%%
#Bar Plot: Distribution of Work Year
plt.figure(figsize=(8, 6))
sns.countplot(x='work_year', data=df, palette='Set2')
plt.title('Work Year Distribution')
plt.xlabel('Work Year')
plt.ylabel('Count')
plt.show()

#Heatmap for Company Size and Salary Relationship
plt.figure(figsize=(10, 6))
salary_by_size = df.groupby('company_size')['salary_in_usd'].mean().reset_index()
sns.heatmap(salary_by_size.set_index('company_size').T, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Average Salary by Company Size')
plt.show()


# Insights: Average Salary by Job Title
average_salary_by_title = df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
print("\nAverage Salary by Job Title (Top 10):")
print(average_salary_by_title.head(10))

# Visualize average salaries by job title
top_roles = average_salary_by_title.head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_roles.values, y=top_roles.index, palette='coolwarm')
plt.title('Average salary by Job Title (Top 10)')
plt.xlabel('Average salary (USD)')
plt.ylabel('job_title')
plt.tight_layout()
plt.show()
# %%
# Top junior data science salaries
junior_salaries = df[df['experience_level'] == 'Entry-level']
top_junior_salaries = junior_salaries.sort_values(by='salary_in_usd', ascending=False).head(10)

print("\nTop 10 Junior Data Science Salaries:")
print(top_junior_salaries[['job_title', 'salary_in_usd', 'company_size', 'company_location']])

# Visualize top junior data science salaries
plt.figure(figsize=(12, 6))
sns.barplot(data=top_junior_salaries, x='salary_in_usd', y='job_title', palette='Spectral')
plt.title('Top 10 Junior Data Science Salaries')
plt.xlabel('Salary (USD)')
plt.ylabel('Job Title')
plt.tight_layout()
plt.show()

#%%
# Analyze which company size pays well for entry-level jobs
avg_salary_by_company_size = junior_salaries.groupby('company_size')['salary_in_usd'].mean().sort_values(ascending=False)

print("\nAverage Salary by Company Size for Entry-Level Jobs:")
print(avg_salary_by_company_size)

# Visualize average salary by company size for entry-level jobs
plt.figure(figsize=(10, 5))
sns.lineplot(x=avg_salary_by_company_size.index, y=avg_salary_by_company_size.values, marker='o', linestyle='-', color='teal')
plt.title('Average Salary by Company Size for Entry-Level Jobs')
plt.xlabel('Company Size')
plt.ylabel('Average Salary (USD)')
plt.tight_layout()
plt.show()

# %%
# Relationship between company size and Data Science salaries
country_us = df[df["company_location"] == "United States"]
small_company = country_us[country_us["company_size"] == "Small"]
medium_company = country_us[country_us["company_size"] == "Medium"]
large_company = country_us[country_us["company_size"] == "Large"]

# Print the rows of each company size category
print(f"Small companies: {small_company.shape[0]} rows")
print(small_company.head())
print(f"Medium companies: {medium_company.shape[0]} rows")
print(medium_company.head())
print(f"Large companies: {large_company.shape[0]} rows")
print(large_company.head())

# Create violin plots for salary distributions across different company sizes.
plt.figure(figsize=(18, 6))

# Plot for small companies
plt.subplot(1, 3, 1)
sns.violinplot(x=small_company["company_size"], y=small_company["salary_in_usd"])
plt.title("Small Company")

# Plot for medium companies
plt.subplot(1, 3, 2)
sns.violinplot(x=medium_company["company_size"], y=medium_company["salary_in_usd"])
plt.title("Medium Company")

# Plot for large companies
plt.subplot(1, 3, 3)
sns.violinplot(x=large_company["company_size"], y=large_company["salary_in_usd"])
plt.title("Large Company")

# Adjust the layout and display the plot in VS Code
plt.tight_layout()
plt.show()

# %%
# Relationship between roles and salaries based on company size
# Define Functions for Reusability
def filter_companies_by_size(df, company_size):
    """
    Filters the dataframe by company size and returns the filtered data.
    """
    return df[df["company_size"] == company_size]

def calculate_average_salary(df, title, company_size):
    """
    Calculate the average salary for a given job title and company size.
    """
    job_title_data = df[df["job_title"] == title]
    avg_salary = job_title_data["salary_in_usd"].mean()
    return {"title": title, "company_size": company_size, "salary": int(avg_salary)}

def get_common_roles(small_company, medium_company, large_company):
    """
    Find the common roles across Small, Medium, and Large companies.
    """
    small_roles = set(small_company["job_title"])
    medium_roles = set(medium_company["job_title"])
    large_roles = set(large_company["job_title"])
    
    # Get intersection of roles
    common_roles = list(small_roles & medium_roles & large_roles)
    return common_roles

def create_role_salary_df(small_company, medium_company, large_company):
    """
    Create a DataFrame with the average salary per role based on company size.
    """
    roles = get_common_roles(small_company, medium_company, large_company)
    role_df = []
    
    for title in roles:
        # Calculate average salary for each role and company size
        role_df.append(calculate_average_salary(small_company, title, "Small"))
        role_df.append(calculate_average_salary(medium_company, title, "Medium"))
        role_df.append(calculate_average_salary(large_company, title, "Large"))
    
    role_df = pd.DataFrame(role_df)
    
    # Sort the DataFrame by salary in descending order
    role_df = role_df.sort_values(by="salary", ascending=False)
    
    return role_df

def visualize_salary_by_role(role_df):
    """
    Visualize the average salary by role based on company size using Matplotlib and Seaborn.
    """
    plt.figure(figsize=(12, 8))

    # Custom color palette for the company sizes
        # Define color mapping for company size
    color_palette = {"Small": "red", "Medium": "blue", "Large": "lightgreen"}

    # Create a barplot of the average salary by role and company size with custom colors
    sns.barplot(x="salary", y="title", hue="company_size", data=role_df, palette=color_palette)
    
    # Customize plot labels and title
    plt.title("Average Salary per Role Based on Company Size")
    plt.xlabel("Average Salary (USD)")
    plt.ylabel("Job Title")
    plt.tight_layout()

    # Show plot
    plt.show()

# Main Execution Flow
def main():
    """
    Main function to execute the structured code.
    """
    # Load the dataset (assuming it's already loaded into df)
    df = pd.read_csv("data_science_salaries.csv")

    # Filter data for United States and company sizes
    united_states_df = df[df["company_location"] == "United States"]
    
    # Filter data by company size
    small_company = filter_companies_by_size(united_states_df, "Small")
    medium_company = filter_companies_by_size(united_states_df, "Medium")
    large_company = filter_companies_by_size(united_states_df, "Large")
    
    # Create the role salary dataframe
    role_df = create_role_salary_df(small_company, medium_company, large_company)
    
    # Visualize the results
    visualize_salary_by_role(role_df)

# Run the Main Function
if __name__ == "__main__":
    main()


# Aniruddh EDA














# %%
# Logistic Regression Model to Predict High Salary Jobs:
# Dependent Variable: High Salary (1 if salary_in_usd > median salary, 0 otherwise)
# Independent Variables: experience_level (encoded), job_title (encoded), company_size (encoded)

# Step 1: Calculate Median Salary and Create Binary Variable
# Calculate median salary
median_salary = df['salary_in_usd'].median()
print("Median Salary:", median_salary)


# Create High Salary binary variable
df['high_salary'] = np.where(df['salary_in_usd'] > median_salary, 1, 0)
print(df[['salary_in_usd', 'high_salary']].head())


# Encode categorical variables
le = LabelEncoder()
df['experience_level_encoded'] = le.fit_transform(df['experience_level'])
df['job_title_encoded'] = le.fit_transform(df['job_title'])
df['company_size_encoded'] = le.fit_transform(df['company_size'])
df['work_models_encoded'] = le.fit_transform(df['work_models'])

print(df[['experience_level', 'experience_level_encoded']].head())
print(df[['job_title', 'job_title_encoded']].head())
print(df[['company_size', 'company_size_encoded']].head())
print(df[['work_models', 'work_models_encoded']].head())

# Define features and target
X = df[['experience_level_encoded', 'job_title_encoded', 'company_size_encoded', 'work_models_encoded']]
y = df['high_salary']
print("Features (X):")
print(X.head())
print("Target (y):")
print(y.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Features (X_train):")
print(X_train.head())
print("Test Features (X_test):")
print(X_test.head())
print("Training Target (y_train):")
print(y_train.head())
print("Test Target (y_test):")
print(y_test.head())



# Step 2: Check Assumptions of Logistic Regression
# 1. Linearity of the Logit:
# We will check this using a method to ensure that there is a linear relationship between the logit of the
# outcome and the continuous independent variables.

# Add a constant to the model
X_train_sm = sm.add_constant(X_train)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Print the summary
print(result.summary())

# Calculate the logit
logit = result.predict(X_train_sm)

# Check for linearity of the logit
import matplotlib.pyplot as plt
import seaborn as sns

for feature in X_train.columns:
    sns.regplot(x=X_train[feature], y=logit, logistic=True)
    plt.xlabel(feature)
    plt.ylabel('Logit')
    plt.show()

# Check for multicollinearity using the Variance Inflation Factor (VIF).
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns

print(vif)

# Visualization of VIF
plt.figure(figsize=(10, 6))
plt.barh(vif["features"], vif["VIF Factor"], color='skyblue')
plt.xlabel("VIF Factor")
plt.title("Variance Inflation Factor")
plt.show()

# Check for high leverage points or influential points using leverage and Cook's distance.
# Add a constant to the model
X_train_sm = sm.add_constant(X_train)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Get influence measures
influence = result.get_influence()
leverage = influence.hat_matrix_diag
cooks = influence.cooks_distance[0]

# Plot leverage vs. Cook's distance
plt.figure(figsize=(10, 6))
plt.scatter(leverage, cooks, alpha=0.5)
plt.xlabel('Leverage')
plt.ylabel("Cook's Distance")
plt.title('Leverage vs. Cook\'s Distance')
plt.show()

# Identify high leverage points
high_leverage_points = np.where(leverage > (2 * (X_train.shape[1] + 1) / X_train.shape[0]))[0]
print("High leverage points:", high_leverage_points)

# To build Model 2 using only significant predictors and incorporating dummy variables for categorical features, follow these steps:

# Data Preparation
# Encode Categorical Variables: Convert categorical variables into dummy variables.
# Select Predictors: Use only significant predictors (experience_level_encoded and job_title_encoded).

# Encode categorical variables
X_train_dummies = pd.get_dummies(X_train[['experience_level_encoded', 'job_title_encoded']], drop_first=True)

# Add a constant term for the intercept
X_train_sm = sm.add_constant(X_train_dummies)

# Build the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Print the summary
print(result.summary())



# Adding Interaction Terms and Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

# Create interaction terms and polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_dummies)

# Add a constant term for the intercept
X_train_poly_sm = sm.add_constant(X_train_poly)

# Build the logistic regression model
logit_model_poly = sm.Logit(y_train, X_train_poly_sm)
result_poly = logit_model_poly.fit()

# Print the summary
print(result_poly.summary())

# LR Assessment 
# Confusion Matrix and classification metrics (accuracy, precision, recall, F1-score, specificity):

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predict probabilities
y_pred_prob = result_poly.predict(X_train_poly_sm)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_train, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy, Precision, Recall, F1-score
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

# Specificity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# ROC-AUC
roc_auc = roc_auc_score(y_train, y_pred_prob)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Specificity: {specificity}")
print(f"ROC-AUC: {roc_auc}")

# McFadden's R2
mcfadden_r2 = 1 - (result_poly.llf / result_poly.llnull)
print(f"McFadden's R2: {mcfadden_r2}")

# visualize the ROC curve to find the threshold
from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob)

# Find the optimal threshold
optimal_threshold = thresholds[np.argmax(tpr - fpr)]

print(f"Optimal Threshold: {optimal_threshold}")

# Use the optimal threshold to predict
y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

# Evaluate performance metrics with the optimal threshold
cm_optimal = confusion_matrix(y_train, y_pred_optimal)
accuracy_optimal = accuracy_score(y_train, y_pred_optimal)
precision_optimal = precision_score(y_train, y_pred_optimal)
recall_optimal = recall_score(y_train, y_pred_optimal)
f1_optimal = f1_score(y_train, y_pred_optimal)
roc_auc_optimal = roc_auc_score(y_train, y_pred_prob)

print(f"Confusion Matrix:\n{cm_optimal}")
print(f"Accuracy: {accuracy_optimal}")
print(f"Precision: {precision_optimal}")
print(f"Recall: {recall_optimal}")
print(f"F1-score: {f1_optimal}")
print(f"ROC-AUC: {roc_auc_optimal}")


# Plot Confusion Matrix for the Polynomial Feature
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Polynomial Feature Model Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Polynomial Feature Model Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

# Model with Optimal Threshold Confusion Matrix
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title('Optimal Threshold Model Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.show()

# Plot ROC Curve for both models
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()