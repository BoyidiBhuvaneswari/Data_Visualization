
import pandas as pd

# Load the dataset
df = pd.read_csv('Titanic.csv')
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

#Bar Chart: Survival Rate by Class (Pclass)
sns.barplot(x='Pclass', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

"""**Insight:** Passengers in higher classes (e.g., first class) had significantly higher survival rates."""

#Bar Chart: Survival Rate by Gender
sns.barplot(x='Sex', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

"""**Insight:** Females had a much higher survival rate compared to males."""

#Scatter Plot: Fare vs. Age by Survival
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Fare vs. Age by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', loc='upper right')
plt.show()

"""**Insight:** Younger passengers and those who paid higher fares were more likely to survive."""

#Countplot: Number of Passengers by Embarkation Port
sns.countplot(x='Embarked', data=df, hue='Survived')
plt.title('Number of Passengers by Embarkation Port and Survival')
plt.xlabel('Embarkation Port')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()

"""**Insight**: Most passengers embarked from Southampton (S), followed by Cherbourg (C)."""

#Violin Plot: Survival Distribution by Age
sns.violinplot(x='Survived', y='Age', data=df, split=True)
plt.title('Survival Distribution by Age')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

"""**Insight:** Children had a higher survival probability compared to adults."""

#Pie Chart: Proportion of Survivors vs. Non-Survivors
df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Did Not Survive', 'Survived'], colors=['lightcoral', 'skyblue'])
plt.title('Proportion of Survivors vs. Non-Survivors')
plt.ylabel('')
plt.show()

"""**insigh**t: Approximately two-thirds of passengers did not survive the Titanic disaster"""

#Histogram: Distribution of Ages
df['Age'].plot.hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

"""**Insight:**Most passengers were aged between 20 and 40 years"""

#Box Plot: Fare Distribution by Class
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

"""**Insight:** First-class passengers paid significantly higher fares than second and third-class passengers."""

#Line Plot: Survival Rate by Family Size
df['FamilySize'] = df['SibSp'] + df['Parch']
family_survival = df.groupby('FamilySize')['Survived'].mean()
family_survival.plot(kind='line', marker='o', color='green')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.grid(True)
plt.show()

"""**Insight:** Smaller family sizes (1-3 members) were associated with higher survival rates."""

# Scatter plot of Age vs. Fare with survival as color
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', palette='coolwarm', edgecolor="w", s=100)
plt.title('Age vs. Fare by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

"""**Insight:** Younger passengers and those who paid higher fares had a greater likelihood of survival, as indicated by the clustering of survivors in the lower-age and higher-fare regions"""

#Heatmap: Correlation Between Numerical Features
# Filter the DataFrame to include only numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Plot the heatmap of correlation between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Titanic Features')
plt.show()

"""**Insight:** Fare and Survived show a positive correlation, indicating that higher fares may be associated with better survival rates, while Pclass and Survived have a negative correlation."""