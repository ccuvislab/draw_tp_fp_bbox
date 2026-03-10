# 基於模型評估並存取預測TP、FP、FN數量及bbox座標，並使其可視化在圖像上。

## 專案說明（這個 Work 在幹嘛？）

本專案的目標是**評估並視覺化物件偵測模型的預測品質**，重點在於比較兩種聯邦學習模型（**FedMA** 與 **FedAvg**）在驗證影像上的預測差異。

### 整體流程

```
[模型評估階段]
FedMPEN 模型推論 (FedMA / FedAvg)
    └─> 修改後的 pascal_voc_evaluation.py
           └─> 對每張驗證影像，計算並儲存:
                ├─ TP (True Positive)  : 模型正確預測到的物件框
                ├─ FP (False Positive) : 模型預測但實際上不存在的物件框
                └─ FN (False Negative) : 實際存在但模型沒偵測到的物件框

[比較分析階段]
tp_fp_score_bb.py
    └─> 讀取 FedMA 與 FedAvg 各自的 count 檔
           └─> 計算每張影像的 (tp_diff + fp_diff) 分數
                └─> 找出差異最大的前 10 張影像 → scores.txt

[視覺化階段]
make_bbox.py
    └─> 讀取 scores.txt 找出前 10 張影像
           └─> 在原始影像上繪製彩色邊界框:
                ├─ 綠色框：TP (True Positive)
                ├─ 紅色框：FP (False Positive)
                └─ 黃色框：FN (False Negative)
                     └─> 輸出對比影像，直觀呈現兩種模型的預測差異
```

### 關鍵輸入與輸出

| 階段 | 輸入 | 輸出 |
|------|------|------|
| 模型評估 | FedMPEN 模型權重、Cityscapes 驗證集 | `FedXXX_tp_fp_fn_count.txt`（各圖 TP/FP/FN 數量）、`FedXXX_tp_fp_fn_bb.txt`（各圖 bbox 座標） |
| 比較分析 | FedMA 與 FedAvg 的 count.txt | `tp_fp_diff.txt`（差異值）、`scores.txt`（排序後分數） |
| 視覺化 | scores.txt、bb.txt、原始影像 | 繪製 TP/FP/FN 邊界框的 JPG 影像 |

### 為什麼這樣做？

透過比較 FedMA 與 FedAvg 在同一張影像上的 TP 與 FP 差異，可以：
- 找出兩種聯邦學習策略預測結果差距最大的影像
- 直觀地看出哪種模型在特定影像上表現更好
- 輔助分析聯邦學習聚合方式對物件偵測效果的影響



**基礎環境設置**
+ torch==2.1.1
+ torchvision==0.16.1

**參考模型** 
+ [FedMPEN(FedMA、FedAvg)](https://github.com/ccuvislab/FedMPEN)

**修改檔案**
+ pascal_voc_evaluation.py(捕捉TP、FP、FN)
+ trainer_sourceonly.py(讓config file參數傳到pascal_voc_evaluation.py)
+ config.py(新定義參數)

**修改後存放位置**
```
FedMPEN
├── configs
│   └── evaluation
│       └── cityeval.yaml
├── pt
│   ├── engine
│   │   ├── pascal_voc_evaluation.py
│   │   └── trainer_sourceonly.py
│   └── config.py
├── train_net_multiTeacher.py
```

**指令執行**
```
python train_net_multiTeacher.py --eval-only --config configs/evaluation/cityeval.yaml MODEL.WEIGHTS output/multi-teacher_skf2c_foggy_sourceonly_FedMAbackbone/model_final.pth
```

**使用操作(Config File)**
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
+ Fed_XXX_tp_fp_fn_count.txt
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/tp_fp_fn_count.png" width="70%" >

+ Fed_XXX_tp_fp_fn_bbox.txt
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/tp_fp_fn_bbox.png" width="70%" >

**比較兩種模型預測差距(tp_fp_score_bb.py)**
+ FedMA和FedAvg\
  透過FedMA與FedAvg的TP、FP分別相減(tp_diff、fp_diff)，找出差異最大(tp_diff + fp_diff)的前十張影像

+ tp_fp_diff.txt
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/tp_fp_diff.png" width="70%" >

+ score.txt
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/score.png" width="70%" >

**可視化影像(make_bbox.py)**
+ FedMA和FedAvg\
  透過score分數最高的前十張影像，分別加入bbox，比較模型在同張影像上的預測成果\
  綠色=True Positive(TP), 紅色=False Positive(FP), 黃色=False Negative(FN)
  
+ FedAvg
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/frankfurt_000001_000538_leftImg8bit_FedAvg.jpg" width="50%" >

+ FedMA
<img src="https://github.com/ccuvislab/draw_tp_fp_bbox/blob/main/Pic/frankfurt_000001_000538_leftImg8bit_FedNA.jpg" width="50%" >


