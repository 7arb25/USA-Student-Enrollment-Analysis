# U.S. Educational Finances Analysis: Uncovering Insights for Equitable Education

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/Intro.jpg)

This project delves into the complexities of U.S. educational finance data to identify trends, disparities, and key factors influencing student enrollment.  The analysis is specifically designed to provide actionable insights for the InvestInMind Foundation, empowering them to make informed decisions and maximize their impact in promoting equitable educational opportunities.

## Project Overview

This repository houses the complete code, data, and analysis pipeline for exploring U.S. educational finance data.  The project's overarching goals are:

* **Comprehensive Data Processing:**  Clean, transform, and prepare raw educational finance data for in-depth analysis.
* **Visual Exploration:**  Create compelling visualizations to illustrate trends in per-student expenditure, funding source allocation, and other key metrics over time and across different regions.
* **Predictive Modeling:**  Develop and evaluate machine learning models to predict student enrollment based on financial, demographic, and regional factors.
* **Feature Importance Analysis:**  Identify the most influential factors driving student enrollment to guide targeted interventions and resource allocation.
* **Actionable Recommendations:**  Generate data-driven recommendations for the InvestInMind Foundation to inform their strategic initiatives and support for underserved communities.

## Dataset

The primary dataset used in this analysis is sourced from [census.gov](https://www.census.gov/programs-surveys/school-finances/data/tables.html)
[nationsreportcard](https://www.nationsreportcard.gov/ndecore/landing). It encompasses financial data related to U.S. public education, including:

* **Financial Metrics:**
    * `FEDERAL_REVENUE`: Federal funding allocated to education.
    * `STATE_REVENUE`: State funding allocated to education.
    * `LOCAL_REVENUE`: Local funding allocated to education (e.g., property taxes).
    * `TOTAL_REVENUE`: The sum of all revenue sources.
    * `TOTAL_EXPENDITURE`: Total spending on education.
    * `INSTRUCTION_EXPENDITURE`: Spending on direct instruction (teacher salaries, materials).
    * `SUPPORT_SERVICES_EXPENDITURE`: Spending on administration, counseling, etc.
    * `CAPITAL_OUTLAY_EXPENDITURE`: Spending on infrastructure and facilities.
    * `OTHER_EXPENDITURE`: Other miscellaneous educational expenditures.
    * `PER_STUDENT_EXPENDITURE`:  Total expenditure divided by student enrollment.
* **Demographic and Contextual Data:**
    * `STATE`: The U.S. state.
    * `YEAR`: The academic year.
    * `ENROLL`: Student enrollment.
    * `REGION`:  **added by me** The geographic region (Northeast, Midwest, South, West).  *(This column is created during data preprocessing based on state information.)*


A detailed description of each column, including data types and units, can be found in [KAGGLE](https://www.kaggle.com/datasets/noriuk/us-educational-finances).  The data spans from [1992] to [2015].

## Methodology

This project employs a multi-faceted analytical approach:

1. **Data Acquisition and Cleaning:**
    * Data is loaded from the specified source.
    * Missing values are handled using appropriate imputation techniques (KNN Imputer).
    * Data types are verified and corrected.
    * State names are standardized for consistency.
    * Duplicate rows are removed.

2. **Exploratory Data Analysis (EDA):**
    * Descriptive statistics (mean, median, standard deviation, percentiles) are calculated for key variables.
    * Visualizations (histograms, box plots, scatter plots, line charts) are generated to explore data distributions, trends over time, and relationships between variables.
    * Correlation analysis is performed to quantify linear relationships between numerical features.

3. **Regional Analysis:**
    * States are grouped into four regions (Northeast, Midwest, South, West).
    * Per-student expenditure, funding source allocation, and enrollment trends are analyzed and visualized for each region.
    * Statistical tests (ANOVA, post-hoc tests) are used to determine if regional differences are statistically significant.

4. **Machine Learning for Enrollment Prediction:**
    * **Feature Engineering:** New features are created, such as per-student expenditure for different expenditure categories, funding source percentages, and year-over-year changes.
    * **Data Preprocessing:** Numerical features are standardized using `StandardScaler`, and categorical features (region) are one-hot encoded using `OneHotEncoder`.
    * **Model Selection:** Two regression models are trained and evaluated:
        * **Linear Regression:** A baseline model to establish a benchmark.
        * **Random Forest Regressor:** A more complex model to capture potential non-linear relationships.
    * **Model Evaluation:** Model performance is assessed using Mean Squared Error (MSE) and R-squared (R²) metrics.
    * **Hyperparameter Tuning (Optional):**  If computational resources allow, model hyperparameters may be tuned using techniques like GridSearchCV or RandomizedSearchCV.
  **WE DIDN'T DO THAT FOR FUNDAMENTAL REASONS**

5. **Feature Importance Analysis:**

  - Using both graphical and numerical summary
  - drow conclusions and recommendations based on that
  

## DATA CLEANING:

- it was just a few missing "numerical"features

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/heatmap1.jpg)
and dealing with it with KNN Imputer

```python
from sklearn.impute import KNNImputer
imputer=KNNImputer()
x_arr=imputer.fit_transform(df[['ENROLL','OTHER_EXPENDITURE']])
df_=pd.DataFrame(x_arr,columns=imputer.get_feature_names_out())
df.drop(['ENROLL','OTHER_EXPENDITURE'],axis=1,inplace=True)
df=pd.concat([df,df_],axis=1)
```

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/heatmap2.jpg)

- drop duplicated rows

  ``` python
  df.drop_duplicates(inplace=True)
  ```


## EDA

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/Screenshot_%D9%A2%D9%A0%D9%A2%D9%A5-%D9%A0%D9%A2-%D9%A0%D9%A8-%D9%A0%D9%A2-%D9%A4%D9%A7-%D9%A2%D9%A5-%D9%A7%D9%A6_40deb401b9ffe8e1df2f1cc5ba480b12.jpg)

### Conclusions

- **Upward Trend:** The most prominent feature is the clear upward trend in per-student expenditure over time. This indicates that, on average, the amount of money spent per student in the given context (likely US public education) has increased consistently from 1995 to 2015.
- **Steady Growth:** The increase appears relatively steady, with no dramatic spikes or drops. The line shows a gradual but consistent rise over the 20-year period.
- **Magnitude of Increase:**  While the exact dollar amount isn't clear without units on the y-axis, we can observe a substantial relative increase.  Expenditures seem to have more than doubled from the starting point in 1995 to the ending point in 2015.
- **Confidence Interval:** The shaded area suggests that there's some uncertainty about the exact values.  The width of the shaded area gives a visual representation of this uncertainty.  It's likely narrower in the earlier and later years, where there might be more data points, and possibly wider in the middle.

----

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/boxplot.jpg)

This report examines per-student expenditure in public education across four regions of the United States: Midwest, Northeast, South, and West. The analysis is based on data that has been processed to calculate per-student expenditure and categorize states into these regions.

- **Northeast Leads in Spending:** The Northeast region exhibits the highest average per-student expenditure, at 13,022.25. This suggests a greater financial investment in education per student compared to other regions.
- **South Has Lowest Spending:** The South region demonstrates the lowest average per-student expenditure, at 8,249.50. This indicates potentially constrained resources allocated to education per student in this region.
- **Midwest and West in Mid-Range:** The Midwest and West regions fall within the mid-range for per-student spending. The Midwest has an average of 9,730.66, while the West has an average of 9,068.29.
- **Significant Regional Disparities:** The difference in per-student spending between the highest-spending (Northeast) and lowest-spending (South) regions is substantial ($4,772.75). This highlights significant financial disparities in educational investment across different regions.
- **Potential Implications:** These spending disparities could potentially lead to variations in educational resources, quality of instruction, and student outcomes across regions. Further investigation is warranted to explore the correlation between per-student expenditure and educational performance metrics.

### RECOMMENDATIONS
  - **Resource Allocation Review:** Policymakers and educational stakeholders in lower-spending regions (especially the South) may need to review resource allocation strategies to ensure equitable educational opportunities for all students.
  - **Performance Evaluation:** Evaluate educational performance metrics (e.g., standardized test scores, graduation rates) in conjunction with per-student expenditure to determine the impact of spending on student outcomes.

----

``` python
import statsmodels.formula.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm

model = sm.ols('PER_STUDENT_EXPENDITURE ~ C(REGION)', data=df).fit()
anova_table = anova_lm(model, typ=2)  # Use anova_lm directly
print("ANOVA Results:\n", anova_table)

```

- **Highly Significant Difference:** The p-value **(PR(>F))** for C(REGION) is extremely small (1.275420e-57, which is essentially 0 for practical purposes).  This means there is a highly statistically significant difference in per-student expenditure between at least two of the regions.  It's extremely unlikely that these differences are due to random chance.
- **Large F-statistic:** The F-statistic (99.274199) is very large, which further supports the conclusion of significant differences between region means.
- **Sum of Squares:** The sum of squares for C(REGION) (3866.406047) being larger than the residual sum of squares (16175.877745) indicates that a substantial portion of the total variability in per-student expenditure is explained by the region differences.

----

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/barplot1.jpg)

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/barplot2.jpg)

- **Regional Variation in Funding Balance:**

   - **West:** The West region shows the highest reliance on Local funding and the lowest reliance on State funding among the four. This suggests a greater role of property taxes or other local revenue sources in funding education in the West.
   - **Northeast:** The Northeast region exhibits a relatively balanced distribution between State and Local funding, although Local funding still predominates.
   - **Midwest and South:** The Midwest and South regions demonstrate a more significant reliance on State funding compared to the West and Northeast.

- **Potential Implications for Equity:**

   - **Local Funding Disparities:** The heavy reliance on local funding, particularly in the West, could lead to disparities in per-student spending within states. Districts with higher property values can generate more local revenue, potentially resulting in wealthier districts having better-funded schools.
   - **State Role in Equalization:** State funding plays a crucial role in mitigating disparities caused by variations in local wealth. The greater reliance on state funding in the Midwest and South may reflect efforts to equalize funding across districts within those regions.

- **Federal Role in Targeted Support:**  While the overall federal contribution is small, federal funding often targets specific needs or student populations (e.g., low-income students, special education). Its impact might be more significant in specific areas or programs.

----

## Machine Learning Modelling

![img alt](https://github.com/7arb25/USA-Student-Enrollment-Analysis/blob/071d92e8b1223111467215725ff316cc88c2db2b/Imgs/featureImportance.jpg)

``` python 

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
```

### Plotting 

``` python
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
```

**summary:** The Random Forest model provides better predictions of enrollment than Linear Regression.  While both models show a good fit based on R-squared, the high MSE values suggest that there are likely other factors influencing enrollment that are not being captured by the current model.  Further investigation, including feature importance analysis and addressing the high MSE, is necessary.

### Model Insights Interpretation


- **Dominant Influence of OTHER_EXPENDITURE:** 
`OTHER_EXPENDITURE` has by far the highest importance.  This suggests that expenditures categorized as "other" (which could include various administrative, operational, or miscellaneous spending) are the strongest predictor of enrollment.
- **Significant Role of CAPITAL_OUTLAY_EXPENDITURE:**
  `CAPITAL_OUTLAY_EXPENDITURE` (spending on infrastructure, buildings, etc.) also plays a substantial role in predicting enrollment.  This makes intuitive sense, as investments in facilities can influence student capacity and attractiveness of schools.

- **Moderate Influence of STATE_REVENUE and SUPPORT_SERVICES_EXPENDITURE:** `STATE_REVENUE` and `SUPPORT_SERVICES_EXPENDITURE` have a moderate impact.  State funding and spending on support services (administration, counseling, etc.) are relevant, but less influential than the top two.
- **Lower Influence of Other Factors:** The remaining features, including instruction expenditure, federal and local revenue, and region, have relatively low importance in comparison.  While they still contribute to the model's predictive power, their influence is less pronounced.'"

----    

## Report and Recommendations:

   
1. **Focus on Expenditure Analysis:** The strong influence of expenditures, particularly `OTHER_EXPENDITURE` and `CAPITAL_OUTLAY_EXPENDITURE`, suggests that a detailed breakdown and analysis of spending patterns are crucial to understanding enrollment trends.  InvestInMind could focus on research that examines:
   
   - What constitutes "OTHER_EXPENDITURE" and how it varies across districts or states.
   - The relationship between capital investments and enrollment growth or decline.


2. **Consider Non-Financial Factors:** While funding sources and expenditures are important, the relatively low importance of "REGION" suggests that other non-financial factors might be at play.  InvestInMind should consider exploring:

   - Demographic changes (population growth, migration patterns).
   - School quality and reputation (test scores, graduation rates, program offerings).
   - Parental choice and school choice programs.
   - Economic conditions and employment opportunities.
3. **Address Data Limitations:** The high MSE values in the machine learning models, despite high R-squared, suggest that there might be data limitations.  InvestInMind could advocate for:

   - Improved data collection and reporting on educational finances.
   - Standardized definitions and categorization of expenditures (especially "OTHER_EXPENDITURE").
   - More granular data (e.g., at the district or school level).
4. **Targeted Support for Underserved Areas:** Given the potential disparities in local funding, InvestInMind could focus its support on:

   - Districts with low LOCAL_REVENUE or high poverty rates.
   - Areas where capital investments are needed to improve facilities and accommodate enrollment growth.
5. **Policy Advocacy: Based on the research and analysis, InvestInMind can advocate for:**

   - More equitable funding formulas at the state and local levels.
   - Policies that promote efficient and effective use of educational resources.
   - Increased transparency and accountability in educational spending.
6. **Further Research:**  Conduct further research to:

   -Explore the causal relationships between expenditures and enrollment (using more advanced statistical techniques).
   - Investigate the impact of specific programs or interventions on enrollment and student outcomes.

     
## Code

The code for this project is written in Python and leverages the following libraries:

* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical computing.
* `matplotlib`: For creating static visualizations.
* `seaborn`: For creating statistically informative and visually appealing visualizations.
* `scikit-learn`: For machine learning tasks (model training, evaluation, preprocessing).
* `statsmodels`: For statistical modeling and hypothesis testing.

The main script for the analysis is educational_finances_analysis.py. Supporting scripts or Jupyter notebooks may be included in the repository for specific tasks or analyses.

## Running the Code

To execute the analysis, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[7arb25]/[USA-Student-Enrollment-Analysis].git
   cd [USA-Student-Enrollment-Analysis]

## Contact

* **Analyst:** [Abdelrahman Harb](3bd0.g0m3aa@gmail.com)
* **LinkedIn:** [link](https://LinkedIn.com/in/3bd0g0m3aa)
* **InvestInMinds Foundation:**
   - [Ava Jerman - `CEO`](investinmindsfoundation@gmail.com)
   - [Rafi Ptashny - `Data Manager`](rafiptashny@gmail.com)
