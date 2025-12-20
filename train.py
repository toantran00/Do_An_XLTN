# train.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import config

def train_and_evaluate(X, y):
    print("--- Bắt đầu chia dữ liệu ---")
    
    # 1. Mã hóa nhãn (Chữ -> Số)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, config.LABEL_ENCODER_PATH) # Lưu bộ mã hóa để dùng lúc Demo
    
    # 2. Chia Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 3. Train Model
    print(f"--- Đang train Random Forest ({config.N_ESTIMATORS} cây)... ---")
    model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Lưu Model
    joblib.dump(model, config.MODEL_PATH)
    print(f">>> Đã lưu model vào {config.MODEL_PATH}")
    
    # 5. Đánh giá
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> ĐỘ CHÍNH XÁC: {acc*100:.2f}%")
    
    print("\nChi tiết:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 6. Vẽ biểu đồ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.show()