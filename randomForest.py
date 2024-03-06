from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tạo dữ liệu giả lập
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, random_state=42)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình
rf_classifier.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = rf_classifier.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình Random Forest: {accuracy:.4f}")
