import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv(r"G:\Exposys_Project\Mall_Customers.csv")
print(df.head())
# print(df.head)
#Violinplot Visualization of age.
plt.figure(figsize=(10, 10))
plt.title("Age Difference")
sns.axes_style("darkgrid")
sns.violinplot(y=df["Age"])
plt.show()
# Scatterplot representation of Age vs Annual Income
plt.figure(figsize=(12, 12))
plt.title("Age vs Annual Income(Scatterplot)")
sns.axes_style("dark")
sns.scatterplot(x=df["Age"], y=df["Annual Income (k$)"])
plt.show()
# Different Boxplots of Annual Income,Spending Score and Age
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.boxplot(y=df["Annual Income (k$)"],color="Green")
plt.subplot(1,3,2)
sns.boxplot(y=df["Spending Score (1-100)"])
plt.subplot(1,3,3)
sns.boxplot(y=df["Age"])
plt.show()
# Barplot of different age group and their count.
age18_30= df.Age[(df.Age>=18) & (df.Age<=30)]
age31_43= df.Age[(df.Age>=31) & (df.Age<=43)]
age44_56= df.Age[(df.Age>=44) & (df.Age<=56)]
age57_above= df.Age[(df.Age>=57)]
X=["18-30","31-43","44-56","57&above"]
Y=[len(age18_30.values),len(age31_43.values),len(age44_56.values),len(age57_above.values)]
plt.figure(figsize=(15, 6))
sns.barplot(x=X,y=Y,palette="crest")
plt.title("Different age groups and their numbers.")
plt.xlabel("Age Groups")
plt.ylabel("Quantity")
plt.show()
genders = df.Gender.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values,width=0.5,palette="rocket_r")
plt.show()

#Finding best number of clusters
from sklearn.cluster import KMeans
wcss = []
for k in range(1,14):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,[3,4]])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,14),wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()
#So we can see that value of 5 will be optimum value for no. of cluster.
kmeans_model = KMeans(n_clusters=5)
kmeans_model.fit_predict(df.iloc[:,[3,4]])
df["Cluster no."] = kmeans_model.labels_
print(df)
km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,[4,5]])
df["label"] = clusters
plt.figure(figsize=(12,8))
plt.title("Kmeans Representation of clusters.")
sns.scatterplot(x = 'Annual Income (k$)',y = 'Spending Score (1-100)',hue="label",  
                 palette=['green','orange','brown','dodgerblue','red'], legend='full',data = df  ,s = 60 )
plt.show()


km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,[4,5]])
df["label"] = clusters

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0],
           c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1],
           c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2],
           c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3],
           c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4],
           c='purple', s=60)
ax.view_init(30, 185)
plt.title("3 dimensional Representation of clusters.")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()
#Showing the customer base after clustering
for k in range(5):
    print(f'Cluster no. : {k}')
    print(df[df.label == k].describe().iloc[[ 0, 1, 2,3], :-1])
    print('\n\n')
