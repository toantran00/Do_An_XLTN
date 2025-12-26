# config.py
import os

# --- CẤU HÌNH ĐƯỜNG DẪN CƠ SỞ (BASE PATH) ---
# Lấy đường dẫn thực tế của file config.py, sau đó lấy thư mục cha của nó (chính là thư mục Project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CÁC THƯ MỤC CON ---
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
TEST_AUDIO_DIR = os.path.join(BASE_DIR, 'test_audio_files')

# --- ĐẢM BẢO THƯ MỤC TỒN TẠI ---
# Nếu chưa có folder model thì tự tạo
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- FILE PATH CỤ THỂ ---
MODEL_PATH = os.path.join(MODEL_DIR, "urban_sound_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CSV_PATH = os.path.join(DATASET_DIR, 'UrbanSound8K.csv')

# --- CẤU HÌNH XỬ LÝ ÂM THANH ---
N_MFCC = 40  
N_ESTIMATORS = 100