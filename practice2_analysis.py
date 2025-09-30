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