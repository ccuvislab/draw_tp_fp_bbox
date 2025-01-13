# 基於模型評估並存取預測TP、FP、FN數量及bbox座標，並使其可視化在圖像上。

**基礎環境設置**
+ torch==2.1.1
+ torchvision==0.16.1

**基準模型** 
+ [FedMPEN](https://github.com/ccuvislab/FedMPEN) (若模型評估的架構中有pascal_voc_evaluation.py，即可進行pascal_voc的評估)

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

**生成範例**

