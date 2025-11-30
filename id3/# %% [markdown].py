# %% [markdown]
# # Import library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # import dataset

# %%
df = pd.read_csv('WineQT.csv')

# %% [markdown]
# # Pisahkan fitur dan target

# %%

X = df.drop(columns=['quality', 'Id']) # hapus kolom id dan quality karena bukan target atau variabel X
y = df['quality']

print("fitur X = ", X)
print("target y =", y)

# %% [markdown]
# #  Split data atau pisahkan data

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("data siap dilatih")

# %% [markdown]
# # Latih model id3 dengan kriteria ENTROPY

# %%

# menggunakan gain = entropy untuk membuat kriteria = entropy
id3_model = DecisionTreeClassifier(
    criterion='entropy',        # Ini kunci ID3
    random_state=42,
    max_depth=None,             # ID3 tidak membatasi kedalaman (tanpa pruning)
    min_samples_split=2,
    min_samples_leaf=1
)

id3_model.fit(X_train, y_train)





# %% [markdown]
# # Prediksi
# 

# %%
y_pred = id3_model.predict(X_test)
print("prediksi = ", y_pred)

# %% [markdown]
# # Akurasi
# 

# %%
akurasi = accuracy_score(y_test, y_pred)
print("akurasi model id3 = ", akurasi)

# %% [markdown]
# # Evaluasi model

# %%
print("hasil klasifikasi")
print(classification_report(y_test, y_pred))

# %% [markdown]
# # Confusion Matrix

# %%
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix – ID3-style Decision Tree')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# %% [markdown]
# # 8. Prediksi data baru (dari id 3 dataset)

# %%
data_baru = np.array([[11.2, 0.28, 0.56, 1.9, 0.075, 17.0, 60.0, 0.998, 3.16, 0.58, 9.8]])

pred = id3_model.predict(data_baru)
pred_proba = id3_model.predict_proba(data_baru)

print(f"data dari id 3 =  Prediksi kualitas = {pred[0]}")
print("peluang per kelas =")
for kelas, prob in zip(id3_model.classes_, pred_proba[0]):
    print(f"  kualitas = {kelas} || peluang = {prob}")

# %% [markdown]
# # simpan model

# %%
with open('id3_data_wine.pkl', 'wb') as f:
    pickle.dump(id3_model, f)

print("model berhasil disimpan")


