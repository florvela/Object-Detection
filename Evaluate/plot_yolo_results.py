import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb
import json

yolov5_models_path = '../Models/yolov5/'
yolov7_models_path = '../Models/yolov7/'
detectron2_models_path = '../Models/detectron2/'

model_names = []
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10))
legends = []

for model_name in os.listdir(yolov5_models_path):
    if "results_1500_epochs_L_No_frozen_32_batch" in model_name or "results_500_epochs_L_10_frozen_32_batch" in model_name:
        df = pd.read_csv(yolov5_models_path + model_name + "/results.csv")
        df = df[:100]
        df.columns = df.columns.str.strip()

        df.plot(x="epoch", y="metrics/precision", kind="line", ax=ax1, title="Precision")
        df.plot(x="epoch", y="metrics/recall", kind="line", ax=ax2, title="Recall")
        df.plot(x="epoch", y="metrics/mAP_0.5", kind="line", ax=ax3, title="mAP_0.5")
        df.plot(x="epoch", y="metrics/mAP_0.5:0.95", kind="line", ax=ax4, title="mAP_0.5:0.95")

        plot_legend = "yolov5_" + model_name.replace("results_500_epochs_","").replace("results_1500_epochs_","")
        legends.append(plot_legend)

for model_name in os.listdir(yolov7_models_path):
    if "results_100" in model_name or "results_1500" in model_name:
        df = pd.read_csv(yolov7_models_path + model_name + "/results.csv")
        df = df[:100]
        df["epoch"] = df["Epoch"].str.replace("/99","")

        df.plot(x="epoch", y="P", kind="line", ax=ax1, title="Precision")
        df.plot(x="epoch", y="R", kind="line", ax=ax2, title="Recall")
        df.plot(x="epoch", y="mAP@.5", kind="line", ax=ax3, title="mAP_0.5")
        df.plot(x="epoch", y="mAP@.5:.95", kind="line", ax=ax4, title="mAP_0.5:0.95")

        plot_legend = "yolov7_" + model_name.replace("results_100_epochs_","")
        legends.append(plot_legend)


# for model_name in os.listdir(detectron2_models_path):
#     if "results_" in model_name:
#         # f = open(detectron2_models_path + model_name + "/metrics.json")
#         # data = json.load(f)
#         # pdb.set_trace()
#         df = pd.read_json(detectron2_models_path + model_name + "/metrics.json")
#
#         df["epoch"] = df["iteration"]
#
#         df.plot(x="epoch", y="P", kind="line", ax=ax1, title="Precision")
#         df.plot(x="epoch", y="R", kind="line", ax=ax2, title="Recall")
#         df.plot(x="epoch", y="mAP@.5", kind="line", ax=ax3, title="mAP_0.5")
#         df.plot(x="epoch", y="mAP@.5:.95", kind="line", ax=ax4, title="mAP_0.5:0.95")

#
ax1.legend(legends, loc=4)
ax2.legend(legends, loc=4)
ax3.legend(legends, loc=4)
ax4.legend(legends, loc=4)

plt.savefig('img/yolo_results_v2.png')
# plt.show()

# pdb.set_trace()
