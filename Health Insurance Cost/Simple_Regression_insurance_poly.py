import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pylab as plt
import numpy as np
################################################
data=pd.read_csv('Linear_Regression/Insurance/insurance.csv')
# For binary categorical variables, replace categories with 0 and 1
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
#data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
# For the 'region' variable, use get_dummies to perform one-hot encoding
#data = pd.get_dummies(data, columns=['region'], drop_first=True)
data = pd.get_dummies(data, columns=['smoker'])
data = pd.get_dummies(data, columns=['region'], drop_first=True)
# Define the features and the target variable
X = data.drop('charges', axis=1)  # Replace 'charges' with the name of your target variable
Y = data['charges']  # Replace 'charges' with the name of your target variable
##########################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#####################

# Determine the layout of the subplots grid
num_features = X_train.shape[1]
num_rows = int(np.ceil(num_features / 3))  # For example, arrange in a grid with 3 columns
num_cols = 3

# Generate a palette with enough colors for each column
palette = sns.color_palette("hsv", num_features)

# Set up the figure size
plt.figure(figsize=(15, num_rows * 5))

for i in range(num_features):
    column_name = X_train.columns[i]  # where 'i' is the column index
    
    # Create a subplot for each feature
    plt.subplot(num_rows, num_cols, i + 1)
    
    # Plot the histogram
    sns.histplot(X_train[column_name], kde=True, color=palette[i])
    
    plt.title(f'Distribution of {column_name}')

# Adjust the layout
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# Show the plot
plt.show()

#############################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly )
X_test_scaled  = scaler.transform(X_test_poly)
###################

##################
model=LinearRegression()
model.fit(X_train_scaled ,Y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error( Y_test,y_pred)
print('mse',mse)


plt.figure()
plt.scatter(Y_test,y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()
