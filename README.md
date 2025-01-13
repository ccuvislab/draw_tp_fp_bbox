# 基於模型評估並存取預測TP、FP、FN數量及bbox座標，並使其可視化在圖像上。

**基礎環境設置**
+ torch==2.1.1
+ torchvision==0.16.1

**基準模型** 
+ [FedMPEN](https://github.com/ccuvislab/FedMPEN)

**修改檔案**
+ pascal_voc_evaluation.py
+ trainer_sourceonly.py

**修改後存放位置**
```
FedMPEN
├── configs
│   └── evaluation
│       └── cityeval.yaml
├── pt
│   └── engine
│       ├── pascal_voc_evaluation.py
│       └── trainer_sourceonly.py
├── train_net_multiTeacher.py
```

**指令執行**
```
python train_net_multiTeacher.py --eval-only --config configs/evaluation/cityeval.yaml MODEL.WEIGHTS output/multi-teacher_skf2c_foggy_sourceonly_FedMAbackbone/model_final.pth
```

**生成操作**
+ SAVE_TP_FP_FN\
  選擇是否存取評估模型對validation影像的TP、FP、FN之數量及bbox
+ STORE_TP_FP_FN_ROOT_PATH\
  可自行定義TP、FP、FN之數量及bbox存放的主路徑
+ STORE_TP_FP_FN_FILE_NAME\
  可根據評估模型的名稱，存取TP、FP、FN之數量及bbox，並統一存放在主路徑底下
+ EVAL_OVTHRESH\
  可根據想要的閾值，篩選TP、FP、FN之數量及bbox。閾值可從50、55、60、65、70、75、80、85、90、95，任選一種為標準
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/config%20file.png" width="70%" >

**生成結果**

<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/tp_fp_fn_count.png" width="70%" >
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/tp_fp_fn_bbox.png" width="70%" >

**比較兩種模型預測差距**
+ FedMA和FedAvg\
  透過FedMA與FedAvg的TP、FP分別相減(FedMA['tp'] - FedAvg['tp'] = tp_diff, FedMA['fp'] - FedAvg['fp'] = fp_diff)，找出差異score最大(tp_diff + fp_diff)的前十張影像
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/tp_fp_diff.png" width="70%" >
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/score.png" width="70%" >

