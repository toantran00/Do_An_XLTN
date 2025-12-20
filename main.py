# main.py
import utils
import train

if __name__ == "__main__":
    print("PROJECT NHẬN DIỆN ÂM THANH ĐÔ THỊ (URBANSOUND8K)")
    
    # Bước 1: Lấy dữ liệu
    X, y = utils.load_data()
    
    # Bước 2: Train và Báo cáo
    train.train_and_evaluate(X, y)