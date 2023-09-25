import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\s+', '_', regex=True)

#nonoresult.csv表示原始的结果图,csv文件在runs/train/exp中
original_results = pd.read_csv("runs/train/exp12/results.csv")
label_original = os.path.dirname("runs/train/exp12/results.csv")
#yesyesresult.csv表示提高后的结果图，csv文件在runs/train/exp中
improved_results = pd.read_csv("runs/visdrone-train/new_double_afpn+bifpn+ca+3pan2-epoch300/results.csv")
label_improve = os.path.dirname("runs/visdrone-train/new_double_afpn+bifpn+ca+3pan2-epoch300/results.csv")

# Clean column names
clean_column_names(original_results)
clean_column_names(improved_results)

# Plot mAP@0.5 curves
plt.figure()
#lable属性为曲线名称，自己可以定义
plt.plot(original_results['metrics/mAP_0.5'], label=label_original)
plt.plot(improved_results['metrics/mAP_0.5'], label=label_improve)
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.legend()
plt.title("mAP@0.5 Comparison")
#plt.savefig("mAP_0.5_comparison.png")

# Plot mAP@0.5:0.95 curves
plt.figure()
plt.plot(original_results['metrics/mAP_0.5:0.95'], label="Original YOLOv5")
plt.plot(improved_results['metrics/mAP_0.5:0.95'], label="Improved YOLOv5")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5:0.95")
plt.legend()
#图的标题
plt.title("mAP@0.5:0.95 Comparison")
plt.show()
#图片名称
#plt.savefig("mAP_0.5_0.95_comparison.png")