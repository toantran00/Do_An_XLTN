# config.py
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Tạo folder model nếu chưa có
if not os.path.exists('model'):
    os.makedirs('model')

MODEL_PATH = os.path.join("model", "urban_sound_model.pkl")
LABEL_ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

# --- CẤU HÌNH XỬ LÝ ÂM THANH ---
N_MFCC = 40  # Số lượng đặc trưng (giữ nguyên giống utils.py)

# --- CẤU HÌNH TRAINING (Cái fen đang thiếu) ---
N_ESTIMATORS = 100  # Số lượng cây trong rừng (Random Forest). 
                    # Fen có thể tăng lên 200 hoặc 300 nếu muốn chính xác hơn.