#       Gerekli Kütüphanelerimiz
##    Binary Classification
import pandas as pd
import numpy as np

# Model ve Eğitim Metotları
from sklearn.model_selection import train_test_split, GridSearchCV  # GridSearchCV: Hiperparametre optimizasyonu için
from sklearn.preprocessing import StandardScaler                   # StandardScaler: Veriyi ölçeklemek için
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, recall_score

# Burada "test" verisi modelin hiç görmediği veridir,
# "train" verisi ise modelin öğrenmesi (eğitim alması) için kullanılır.
# Model dediğimiz şey ise verideki örüntüleri bulmaya ve yeni verileri tahmin etmeye yarayan matematiksel bir araçtır.

# Veri Setini Yükleme
file_path = "Data/data.csv"  # Veri setinin dosya yolu
data = pd.read_csv(file_path)

# Veri Setini Kontrol Etme
print("Veri Setinin İlk Satırları:")
print(data.head())

# Özellikler (X) ve Hedef Değişken (y) Ayrımı
# ID sütunu tahmin için gerekli olmadığı için çıkarıyorum.
# Son sütun 'class' hedef değişken olduğu için 'y' oluyor.
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Sınıfı Sayısal Değere Dönüştürme
# "P" 1 olarak kodlansın, "H" 0 olarak kodlansın.
y = y.map({'P': 1, 'H': 0})

# Veriyi Ölçeklendirme
# Bu adım her zaman zorunlu değildir ama performans artışı sağlayabilir.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri Setini Eğitim ve Test Olarak Bölme
# test_size=0.2: Verinin %20'si test için ayrılıyor.
# random_state=42: Rastgelelik kontrolü için sabit bir değer.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hiperparametre Arama
# Burada rastgele değerler deneyeceğiz.
# n_estimators: Orman içinde kaç tane ağaç olacağı
# max_depth: Ağaçların maksimum derinliği
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20]
}

# Özel bir skor fonksiyonu oluşturmak istersek ekleyebilirdik.
# Örneğin duyarlılığı (recall) optimize etmek istiyorsak:
scorer = make_scorer(recall_score)

# GridSearchCV ile modelin en iyi hiperparametrelerini bulalım
# cv=5: 5 katlı çapraz doğrulama
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid,
                           scoring=scorer,
                           cv=5,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# En iyi modeli ekrana bastıralım
print("En İyi Hiperparametreler:", grid_search.best_params_)

# Test verisinde tahmin yapma
y_pred = best_model.predict(X_test)

# Performans Metrikleri: Confusion Matrix, Sensitivity ve Specificity
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix'ten Değerler
tp = cm[1, 1]  # True Positives
tn = cm[0, 0]  # True Negatives
fp = cm[0, 1]  # False Positives
fn = cm[1, 0]  # False Negatives

# Sensitivity (Duyarlılık) ve Specificity (Özgüllük) Hesaplama
# Sensitivity: Pozitif sınıfın (1) ne kadar iyi tahmin edildiğini gösterir.
# Specificity: Negatif sınıfın (0) ne kadar iyi tahmin edildiğini gösterir.
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Sonuçları Yazdırma
print("Confusion Matrix:")
print(cm)
print(f"Sensitivity (Duyarlılık): {sensitivity}")
print(f"Specificity (Özgüllük): {specificity}")

# Ek Performans: Doğruluk (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Doğruluk): {accuracy}")



