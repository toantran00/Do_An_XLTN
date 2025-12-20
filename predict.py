# predict.py
import joblib
import os
import numpy as np
import config
import utils

# --- HÀM DỰ ĐOÁN 1 FILE ---
def predict_single_file(file_path, model, le):
    # 1. Trích xuất đặc trưng
    feature = utils.extract_features(file_path)
    
    if feature is not None:
        # Reshape dữ liệu (1 dòng, nhiều cột)
        feature = feature.reshape(1, -1)
        
        # 2. Dự đoán (ra số)
        pred_index = model.predict(feature)[0]
        
        # 3. Dịch số ra chữ
        label = le.inverse_transform([pred_index])[0]
        
        # 4. Tính độ tự tin (Confidence)
        probs = model.predict_proba(feature)[0]
        confidence = probs[pred_index] * 100
        
        return label, confidence
    else:
        return None, None

# --- HÀM CHẠY CHÍNH ---
def main():
    print("="*40)
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH DỰ ĐOÁN")
    print("="*40)

    # 1. Load Model & Label Encoder (Chỉ load 1 lần duy nhất ở đây)
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.LABEL_ENCODER_PATH):
        print(f"❌ LỖI: Không tìm thấy file model tại: {config.MODEL_PATH}")
        return

    try:
        print("⏳ Đang tải model...")
        model = joblib.load(config.MODEL_PATH)
        le = joblib.load(config.LABEL_ENCODER_PATH)
        print("✅ Load model thành công!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # 2. Cấu hình kiểm tra
    # Bạn muốn test 1 file hay cả folder thì sửa ở đây:
    TEST_MODE = 'FOLDER'  # Chọn 'FILE' hoặc 'FOLDER'
    
    PATH_TO_CHECK = './test_audio_files/gun_shot' # Đường dẫn folder hoặc file
    
    # 3. Thực thi
    if TEST_MODE == 'FOLDER':
        if not os.path.exists(PATH_TO_CHECK):
            print(f"❌ Không tìm thấy thư mục: {PATH_TO_CHECK}")
            return
            
        print(f"\n📂 Đang kiểm tra thư mục: {PATH_TO_CHECK}")
        files = [f for f in os.listdir(PATH_TO_CHECK) if f.endswith('.wav')]
        
        print(f"{'FILENAME':<30} | {'PREDICTION':<20} | {'CONFIDENCE'}")
        print("-" * 70)
        
        for file_name in files:
            full_path = os.path.join(PATH_TO_CHECK, file_name)
            label, conf = predict_single_file(full_path, model, le)
            
            if label:
                print(f"{file_name:<30} | {label.upper():<20} | {conf:.1f}%")
            else:
                print(f"{file_name:<30} | ❌ Lỗi đọc file")
                
    else: # Chế độ test 1 file lẻ
        print(f"\n🎤 Đang kiểm tra file: {PATH_TO_CHECK}")
        label, conf = predict_single_file(PATH_TO_CHECK, model, le)
        if label:
             print(f"\n✅ KẾT QUẢ: 👉 {label.upper()} (Độ tin cậy: {conf:.2f}%)")

if __name__ == "__main__":
    main()