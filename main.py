# main.py
import utils
import train

if __name__ == "__main__":
    print("PROJECT NHẬN DIỆN ÂM THANH ĐÔ THỊ (URBANSOUND8K)")
    
    X, y = utils.load_data()
    
    train.train_and_evaluate(X, y)