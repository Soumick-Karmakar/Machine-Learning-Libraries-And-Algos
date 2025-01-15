import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Datasets:
#=============
#tips = sns.load_dataset('tips')
#print(tips.head())

#flights = sns.load_dataset('flights')
#print(flights.head())

iris = sns.load_dataset('iris')
#print(iris.head())


# Distribution Plots:
#=====================


# Histograms
#sns.histplot(tips['total_bill'], kde=True)
#sns.histplot(tips['total_bill'], kde=False, bins=40)
#sns.kdeplot(tips['total_bill'])
#plt.show()
# kde : Kernel Density Estimation
# kde : shows the line graph on top of the bar graph
# bins: shows the data sets in a more accurate manner with thinner bar graphs

# Scatter Plot
#sns.jointplot(x='total_bill',y='tip',data=tips,kind='resid')
#plt.show()
# kind : this allows us to create different kinds of scatter plots. By default kind = 'scatter'
# kind types : hex - hexagon, reg - regration, kde - contour lines

# Pair Plot
#sns.pairplot(tips, hue='sex',palette='coolwarm')
#plt.show()
# hue : this will take the column name (columns that have categorical data like yes/no, male/female, 0/1, etc) as a parameter and will show the data points in differnt colors
# palette : this will color the entire plot with some different color pattern

# Rugplot 
#sns.rugplot(tips['total_bill'])
#plt.show()


# Categorical plots:
#=====================


# Barplot
#sns.barplot(x='sex',y='total_bill', data=tips, estimator=np.std)
#plt.show()

# Countplot
#sns.countplot(x='sex', data=tips)
#plt.show()

# Boxplot
#sns.boxplot(x='day',y='total_bill', data=tips, hue='smoker')
#plt.show()

# Violinplot
#sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
#plt.show()

# Strip plot
#sns.stripplot(x='day', y='total_bill', data=tips, jitter=True, hue='sex')
#plt.show()

# Swarmplot
#sns.swarmplot(x='day', y='total_bill', data=tips)
#plt.show()

# Combining violin plot and swarm plot
#sns.violinplot(x='day', y='total_bill', data=tips)
#sns.swarmplot(x='day', y='total_bill', data=tips)
#plt.show()


# Matrix Plots:
#================


# Heat map
#tips = tips.select_dtypes(include=['float64', 'int64'])  # this is to filter out only the numerical data from the data set
#tc = tips.corr()
#print(tc)
#sns.heatmap(tc, annot=True, cmap='coolwarm')
#plt.show()
# annot : this will display the gradient values along with the gradient colors
# cmap : this will color the heat map with different colors

#fp = flights.pivot_table(index='month', columns='year', values='passengers')   # using pivot_table
#print(fp)
#sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=0.5)
#plt.show()
# cmap : to chnage the themes (coolwarm, magma, cool)
# linecolor : to seperate each unit of the heatmap with a particular color
# linewidths : to seperate each unit of the heartmap with a line of particular width

# Cluster map
#fp = flights.pivot_table(index='month', columns='year', values='passengers')
#sns.clustermap(fp, cmap='magma', standard_scale=1)
#plt.show()
# standard_scale : this is used for normalization


# Grids:
#================


print(iris['species'].unique())
sns.pairplot(iris)
plt.show()


