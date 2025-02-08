import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/us-educational-finances/states.csv")
df.head()
df.info()
df.describe().T
from sklearn.impute import KNNImputer
imputer=KNNImputer()
x_arr=imputer.fit_transform(df[['ENROLL','OTHER_EXPENDITURE']])
df_=pd.DataFrame(x_arr,columns=imputer.get_feature_names_out())
df.drop(['ENROLL','OTHER_EXPENDITURE'],axis=1,inplace=True)
df=pd.concat([df,df_],axis=1)
df.head()

df.drop_duplicates(inplace=True)
df['PER_STUDENT_EXPENDITURE'] = df['TOTAL_EXPENDITURE'] / df['ENROLL']
#Calculate the percentage change in key metrics
df['ENROLL_CHANGE'] = df['ENROLL'].pct_change()
# Plot trends over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='YEAR', y='PER_STUDENT_EXPENDITURE', data=df)
plt.title('Per-Student Expenditure Over Time')
plt.show()

# Analyze spending allocation
instruction_percentage = df['INSTRUCTION_EXPENDITURE'].sum() / df['TOTAL_EXPENDITURE'].sum()
print(f"Percentage of spending on instruction: {instruction_percentage}")
Percentage of spending on instruction: 0.5179106112726778
df['STATE'].value_counts()

state_to_region = {
    "Alabama": "South",
    "Alaska": "West",
    "Arizona": "West",
    "Arkansas": "South",
    "California": "West",
    "Colorado": "West",
    "Connecticut": "Northeast",
    "Delaware": "Northeast",
    "Florida": "South",
    "Georgia": "South",
    "Hawaii": "West",
    "Idaho": "West",
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Iowa": "Midwest",
    "Kansas": "Midwest",
    "Kentucky": "South",
    "Louisiana": "South",
    "Maine": "Northeast",
    "Maryland": "Northeast",
    "Massachusetts": "Northeast",
    "Michigan": "Midwest",
    "Minnesota": "Midwest",
    "Mississippi": "South",
    "Missouri": "Midwest",
    "Montana": "West",
    "Nebraska": "Midwest",
    "Nevada": "West",
    "New Hampshire": "Northeast",
    "New Jersey": "Northeast",
    "New Mexico": "West",
    "New York": "Northeast",
    "North Carolina": "South",
    "North Dakota": "Midwest",
    "Ohio": "Midwest",
    "Oklahoma": "South",
    "Oregon": "West",
    "Pennsylvania": "Northeast",
    "Rhode Island": "Northeast",
    "South Carolina": "South",
    "South Dakota": "Midwest",
    "Tennessee": "South",
    "Texas": "South",
    "Utah": "West",
    "Vermont": "Northeast",
    "Virginia": "South",
    "Washington": "West",
    "West Virginia": "South",
    "Wisconsin": "Midwest",
    "Wyoming": "West"
}

df['REGION'] = df['STATE'].map(state_to_region) 




plt.figure(figsize=(10, 6))
sns.boxplot(x='REGION', y='PER_STUDENT_EXPENDITURE', data=df)
plt.title('Per-Student Expenditure by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(df.groupby('REGION')['PER_STUDENT_EXPENDITURE'].mean())

import statsmodels.formula.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
model = sm.ols('PER_STUDENT_EXPENDITURE ~ C(REGION)', data=df).fit()
anova_table = anova_lm(model, typ=2)  # Use anova_lm directly
print("ANOVA Results:\n", anova_table)

# Calculate the percentage of funding from each source for each region
funding_by_region = df.groupby('REGION')[['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']].sum()
funding_by_region['TOTAL'] = funding_by_region.sum(axis=1)  # Calculate total revenue for percentages
for col in ['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']:
    funding_by_region[col] = (funding_by_region[col] / funding_by_region['TOTAL']) * 100

print("Funding Source Breakdown by Region:\n", funding_by_region)

# Plotting (example - stacked bar chart)
funding_by_region[['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Funding Source Breakdown by Region')
plt.ylabel('Percentage of Total Funding')
plt.show()

regional_funding = df.groupby('REGION')[['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE', 'TOTAL_EXPENDITURE']].sum()


for source in ['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']:
    regional_funding[source + '_PERCENTAGE'] = (regional_funding[source] / regional_funding['TOTAL_EXPENDITURE']) * 100


regional_funding[['FEDERAL_REVENUE_PERCENTAGE', 'STATE_REVENUE_PERCENTAGE', 'LOCAL_REVENUE_PERCENTAGE']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Funding Source Breakdown by Region')
plt.ylabel('Percentage of Total Expenditure')
plt.show()

print(regional_funding)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


X = df.drop('ENROLL', axis=1) 
y = df['ENROLL']


numerical_features = ['FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE', 'TOTAL_EXPENDITURE', 'INSTRUCTION_EXPENDITURE', 'SUPPORT_SERVICES_EXPENDITURE', 'OTHER_EXPENDITURE', 'CAPITAL_OUTLAY_EXPENDITURE']  # Add other numerical features
categorical_features = ['REGION']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust test size as needed
# Linear Regression Model
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)

# Random Forest Regressor Model
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))])  # Add hyperparameters if needed

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse}")
    print(f"{model_name} - R-squared: {r2}")
    return mse, r2

# Evaluate models
mse_lr, r2_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")
mse_rf, r2_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

# Get feature importances (after fitting the Random Forest model)
feature_importances = pipeline_rf.named_steps['regressor'].feature_importances_


feature_names = pipeline_rf.named_steps['preprocessor'].get_feature_names_out()


importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances (Random Forest):\n", importance_df)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()  # Invert to show most important at the top
plt.tight_layout()
plt.show()
