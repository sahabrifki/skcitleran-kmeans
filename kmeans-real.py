import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns
import csv
from sklearn.cluster import KMeans


data = pd.read_csv('data_mentah_normal.csv')
f1 = data['atribut_ttl'].values
f2 = data['atribut_ack'].values
f3 = data['atribut_windows_value'].values

X = np.array(list(zip(f1, f2, f3)))
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
nomorCluster = kmeans.labels_


df = pd.DataFrame(X)
df.columns = ["ttl", "ack", "windowsvalue"]
df["cluster"] = nomorCluster
colors = sns.color_palette()[0:2]
df = df.sort_values("cluster")
namaCluster = {"0": "Pola_Serangan", "1": "Pola_Normal"}
df["hasilCluster"] = [namaCluster[str(i)] for i in df.cluster]

df.to_csv("cluster_data_normal.csv")