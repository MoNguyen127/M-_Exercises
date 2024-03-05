from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tạo dữ liệu mẫu
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo một bộ phân loại AdaBoost
ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)

# Huấn luyện bộ phân loại
ada_boost.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = ada_boost.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của AdaBoost là:", accuracy)
