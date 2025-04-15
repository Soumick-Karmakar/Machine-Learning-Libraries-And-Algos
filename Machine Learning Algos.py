
# house pricing datasets
from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset['Price']=df.target
#print(dataset.head())

# Dividing the daset into dependent and independent features
X = dataset.iloc[:, :-1] #independent features (here I am taking all the columns except the last one)
y = dataset.iloc[:,-1] #dependent features (here I am taking only the last column)


#####################################################################################################################
########################################## SUPERVISED MACHINE LEARNING ##############################################
#####################################################################################################################

########################### Linear Regression ########################

#link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
### Libraries that support linear regression: stats, skypy, etc.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
#print(mean_mse)  # if the value is less, then the model is good, if the value is more, then the model is bad. This value should go towards zero.



########################### Ridge Regression #########################

#link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
### Libraries that support Ridge regression: stats, skypy, etc.
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning

ridge=Ridge()  # Ridge regression tries to reduce overfitting

params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)
#print(ridge_regressor.best_params_)
#print(ridge_regressor.best_score_)  # if the value is less, then the model is good, if the value is more, then the model is bad. This value should go towards zero.



########################### Lasso Regression #########################

#link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
### Libraries that support Lasso regression: stats, skypy, etc.
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning

lasso=Lasso()  # Lasso regression tries to reduce overfitting

params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)
#print(lasso_regressor.best_params_)
#print(lasso_regressor.best_score_)  # if the value is less, then the model is good, if the value is more, then the model is bad. This value should go towards zero.



# to use x-training and y-training data, we can use the following code
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# in all the above models, we can replace X and y with X_train and y_train
# after the training is done, we can use the model to predict the values of X_test and compare it with y_test to check the accuracy of the model.
lasso_regressor.fit(X_train,y_train)
y_pred = lasso_regressor.predict(X_test)  # this will give the predicted values of y_test
r2_score1 = r2_score(y_test, y_pred)  # this will give the accuracy of the model
#print(r2_score1)



########################### Logistic Regression #########################

#link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
### Libraries that support Logistic regression: stats, skypy, etc.
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
df = load_breast_cancer()
X=pd.DataFrame(df['data'],columns=df['feature_names']) #Independent features
y=pd.DataFrame(df['target'],columns=['Cancer']) #Dependent features

#To check is our dataset is balanced or not
#print(y['Cancer'].value_counts())
#sns.countplot(y['Cancer'])
#plt.show()

params = [ {'C': [ 1, 5, 10], 'max_iter': [100, 150]}]
model1 = LogisticRegression(C=100, max_iter=150)
model = GridSearchCV(model1, param_grid=params, scoring='f1', cv=5)
#model.fit(X_train, y_train)
#print(model.best_params_)
#print(model.best_score_)
#y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#print(accuracy_score(y_test, y_pred))


########################### Decision Tree #########################
#link: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(iris.data, iris.target)
# plt.figure(figsize=(15, 10))
# tree.plot_tree(classifier, filled=True)



#####################################################################################################################
########################################## UNSUPERVISED MACHINE LEARNING ############################################
#####################################################################################################################

########################### K-Means Clustering #########################
#link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html    
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm

# Generating sample data from make_blobs
# This particuatr setting has one distinct cluster and 3 clusters placed close together
X, y = make_blobs(n_samples=500, 
                  n_features=2, 
                  centers=4, 
                  cluster_std=1,  
                  shuffle=True, 
                  random_state=1) # for reproducibility

range_n_clusters = [2, 3, 4, 5, 6]

wcss=[]
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# Now we'll use Silhoutte to validate whether the result we got from Elbow method is correct or not
# The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
# The silhouette score ranges from -1 to 1, where a value close to 1 indicates that the sample is far away from the neighboring clusters.
# A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
# A negative value indicates that the sample might have been assigned to the wrong cluster.

cluster = KMeans(n_clusters=4, random_state=10)
cluster_labels = cluster.fit_predict(X)
# 

########################### Silhouette Score #########################
#link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
from sklearn.metrics import silhouette_samples, silhouette_score

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

# plt.show()

# In the graph, we should see that whether any graph is going less than 0 or not. If so we should avoud that clustering, even if the silhouette value is close to 1.
# the dotted line on the graph is my score.
# Now if we have multiple scores for different numbers of clustering and all the scores are good, then go for the one with high clustering value (n_clusters), as they will give the more generalized model.


