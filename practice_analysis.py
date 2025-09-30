import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# Generate synthetic data
np.random.seed(42)
N = 100
X = np.random.normal(10, 2, N)
y = 3 * X + np.random.normal(0, 5, N)

# Put into a DataFrame
df = pd.DataFrame({'X': X, 'y': y})
print(df.head())
sns.scatterplot(x='X', y='y', data=df)
plt.title('Scatterplot of X vs y')
plt.show()
sns.scatterplot(x='X', y='y', data=df)
plt.title('Scatterplot of X vs y')
plt.show()
X_with_const = sm.add_constant(df['X'])
model = sm.OLS(df['y'], X_with_const).fit()
print(model.summary())
plt.figure(figsize=(8,5))
sns.scatterplot(x='X', y='y', data=df, label='Data')
plt.plot(df['X'], model.predict(X_with_const), color='red', label='Regression Line')
plt.legend()
plt.title('Regression Line Fit')
plt.show()
df.to_csv('synthetic_data.csv', index=False)
print("Data saved to synthetic_data.csv")
# Histogram of X and y
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df['X'], kde=True)
plt.title('Distribution of X')
plt.subplot(1,2,2)
sns.histplot(df['y'], kde=True)
plt.title('Distribution of y')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df)
plt.suptitle('Pairplot of X and y', y=1.02)
plt.show()

