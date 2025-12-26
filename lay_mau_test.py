import os
import shutil
import pandas as pd

# --- CẤU HÌNH ---
DATASET_PATH = './dataset'
CSV_PATH = os.path.join(DATASET_PATH, 'UrbanSound8K.csv')
TEST_DIR = './test_audio_files' 

# Danh sách đủ 10 loại để test model "Full Power"
CLASSES_TO_GET = [
    'air_conditioner', 
    'car_horn', 
    'children_playing', 
    'dog_bark', 
    'drilling', 
    'engine_idling', 
    'gun_shot', 
    'jackhammer', 
    'siren', 
    'street_music'
]

# Số lượng file muốn lấy mỗi loại (Ví dụ: 20 file)
SAMPLES_PER_CLASS = 20

def prepare_test_data():
    print(f"--- 🚀 BẮT ĐẦU TẠO DỮ LIỆU TEST (Full 10 loại) ---")
    
    # 1. Kiểm tra file CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {CSV_PATH}")
        return

    # 2. Dọn dẹp thư mục cũ
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        print(f"🧹 Đã dọn dẹp thư mục test cũ.")
    
    os.makedirs(TEST_DIR)
    
    # 3. Đọc CSV
    print("⏳ Đang đọc danh sách file...")
    df = pd.read_csv(CSV_PATH)
    
    total_copied = 0
    
    # 4. Duyệt và copy
    for label in CLASSES_TO_GET:
        label_dir = os.path.join(TEST_DIR, label)
        os.makedirs(label_dir)
        
        class_df = df[df['class'] == label]
        
        # Lấy ngẫu nhiên
        n_samples = min(SAMPLES_PER_CLASS, len(class_df))
        samples = class_df.sample(n=n_samples)
        
        print(f"   📂 {label}: Đang lấy {n_samples} file...", end=" ")
        
        count_ok = 0
        for _, row in samples.iterrows():
            filename = row['slice_file_name']
            fold = row['fold']
            src_path = os.path.join(DATASET_PATH, f"fold{fold}", filename)
            dst_path = os.path.join(label_dir, filename)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                count_ok += 1
                total_copied += 1
        
        print(f"✅ Xong ({count_ok} file)")

    print("="*40)
    print(f"🎉 ĐÃ XONG! Tổng cộng có {total_copied} file trong '{TEST_DIR}'.")
    print(f"👉 Fen vào file 'predict.py' đổi đường dẫn để test từng loại nhé!")

if __name__ == "__main__":
    prepare_test_data()