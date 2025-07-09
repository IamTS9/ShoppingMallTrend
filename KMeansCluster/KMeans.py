#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataiku
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import io
import json


# In[2]:


dataset = dataiku.Dataset("Mall_Customers_Data_Set_prepared")
df = dataset.get_dataframe()


# In[3]:


df.head()


# In[4]:


features = ['Male', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


inertias = []
ks = list(range(2, 21))


# In[7]:


for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)


# In[8]:


plt.figure(figsize=(10,6))
plt.plot(ks, inertias, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# In[9]:


optimal_k = 10 
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
final_kmeans.fit(X_scaled)


df['Cluster'] = final_kmeans.predict(X_scaled)


# In[10]:


folder = dataiku.Folder("Folder1")

# Save KMeans model
model_buffer = io.BytesIO()
joblib.dump(final_kmeans, model_buffer)
model_buffer.seek(0)
folder.upload_data("kmeans_model.joblib", model_buffer.getvalue())
print("Joblib saved")


# In[12]:


scaler_buffer = io.BytesIO()
joblib.dump(scaler, scaler_buffer)
scaler_buffer.seek(0)
folder.upload_data("scaler.joblib", scaler_buffer.getvalue())
print("Scaler buffer saved")


# In[13]:


df


# In[28]:


import dataiku
import pandas as pd
from io import BytesIO

# Create a BytesIO stream
buffer = BytesIO()
df.to_csv(buffer, index=False)

# Reset stream position to the beginning
buffer.seek(0)

# Write to Dataiku folder
folder = dataiku.Folder("Folder1")
with folder.get_writer("mall_clustered_data.csv") as writer:
    writer.write(buffer.read())


# In[35]:


import matplotlib.pyplot as plt


# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example: Load clustered data from Dataiku folder or local CSV
# df = pd.read_csv("mall_clustered_data.csv")  # or use Dataiku's folder.read()

# Group by cluster and compute means
cluster_means = df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Male"]].mean()

# Convert to numpy array for plotting
data = cluster_means.values
features = cluster_means.columns.tolist()
clusters = cluster_means.index.tolist()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(data.T, cmap='coolwarm', aspect='auto')

# Axis setup
ax.set_xticks(np.arange(len(clusters)))
ax.set_yticks(np.arange(len(features)))
ax.set_xticklabels([f"Cluster {c}" for c in clusters])
ax.set_yticklabels(features)

# Rotate x-labels if needed
plt.xticks(rotation=45)

# Annotate cells
for i in range(len(features)):
    for j in range(len(clusters)):
        value = round(data[j][i], 1)
        ax.text(j, i, str(value), ha='center', va='center', color='black')

# Add colorbar
fig.colorbar(cax)

# Title
plt.title("Heatmap of Feature Averages per Cluster")
plt.tight_layout()
plt.show()


# In[ ]:




