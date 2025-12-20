# utils.py
import librosa
import numpy as np
import os
import pandas as pd # Cần import pandas để đọc file CSV
import config

# 1. Hàm trích xuất đặc trưng (Giữ nguyên)
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config.N_MFCC)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"⚠️ Lỗi file {file_path}: {e}")
        return None

# 2. Hàm Load Data (Phiên bản mới - Đọc từ CSV)
def load_data(dataset_path='./dataset'):
    print("\n" + "="*40)
    print(f"📂 ĐANG LOAD DỮ LIỆU TỪ CSV: {dataset_path}")
    print("="*40)

    # Đường dẫn đến file CSV
    csv_path = os.path.join(dataset_path, 'UrbanSound8K.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ LỖI: Không tìm thấy file {csv_path}")
        return np.array([]), np.array([])

    # Đọc file CSV
    metadata = pd.read_csv(csv_path)
    
    # Danh sách nhãn cần lấy
    target_classes = ['children_playing', 'dog_bark', 'drilling', 'gun_shot']
    
    # Lọc chỉ lấy những dòng thuộc 4 nhãn trên
    filtered_data = metadata[metadata['class'].isin(target_classes)]
    
    features = []
    labels = []
    
    total_files = len(filtered_data)
    processed = 0
    
    print(f"🔍 Tìm thấy {total_files} file phù hợp trong CSV.")
    print("⏳ Đang xử lý âm thanh (sẽ hơi lâu chút nha)...")

    # Duyệt qua từng dòng trong file CSV đã lọc
    for index, row in filtered_data.iterrows():
        file_name = row['slice_file_name']
        fold_num = row['fold']
        label = row['class']
        
        # Tạo đường dẫn: dataset/fold1/100263-2-0-3.wav
        folder_name = f"fold{fold_num}"
        file_path = os.path.join(dataset_path, folder_name, file_name)
        
        data = extract_features(file_path)
        
        if data is not None:
            features.append(data)
            labels.append(label)
        
        # In tiến độ cứ mỗi 100 file
        processed += 1
        if processed % 100 == 0:
            print(f"\r👉 Đã xong: {processed}/{total_files} files", end="")

    print(f"\n✅ HOÀN TẤT! Tổng cộng load được: {len(features)} mẫu.")
    
    return np.array(features), np.array(labels)