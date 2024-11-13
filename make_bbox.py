import os
import cv2

top_10_img_path = "/home/u1755025/FedMPEN_mycode/top10_img"
output_score_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp/scores.txt"
FedMA_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp/FedMA_tp_fp_bb.txt"
FedAvg_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp/FedAvg_tp_fp_bb.txt"

#用來儲存top 10相差最大的照片名稱
top_10_img = []

#找出top 10相差最大的照片(分數最高的top 10)
with open(output_score_file_path, "r") as file:
    image_score = []
    for line in file:
        parts = line.split(": score = ")
        img_name = parts[0].strip()
        score = int(parts[1].strip())
        
        image_score.append((img_name, score))

#依照score由高到低排列
image_score.sort(key = lambda x: x[1], reverse = True)
top_10_img = [image_score[i][0] for i in range(min(10, len(image_score)))]

result = {}

#針對FedMA或FedAvg，做標框動作
with open(FedMA_file_path, "r") as file:
    current_image = None
    for line in file:
        #若bb.txt中存在top 10照片名稱(基本一定存在)，分別對該照片，取出tp及fp bbox
        if any(image_name in line for image_name in top_10_img):
            current_image = line.strip().split(": ")[0]
            tp_content = eval(line.split("tp = ")[1].split(", fp = ")[0].strip())
            fp_content = eval(line.split("fp = ")[1].strip())

            image_files = os.listdir(top_10_img_path)
            #從特地路徑抓取top 10照片來框取
            if current_image + ".jpg" in image_files:
                image_path = os.path.join(top_10_img_path, current_image + ".jpg")
                image = cv2.imread(image_path)
                if image is None:
                    print("圖片無法讀取")
                else:
                    #tp的框使用，綠色
                    for tp_bb in tp_content:
                        #使用前面提取的tp bbox
                        x_min, y_min, x_max, y_max = map(int, tp_bb)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    #fp的框使用，紅色
                    for fp_bb in fp_content:
                        #使用前面提取的fp bbox
                        x_min, y_min, x_max, y_max = map(int, fp_bb)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
                    #框好照片儲存在FedMA或FedAvg的file底下
                    output_image_dir = "/home/u1755025/FedMPEN_mycode/output/mark_FedMA_bb"
                    if not os.path.exists(output_image_dir):
                        os.makedirs(output_image_dir)
                    mark_img = os.path.join(output_image_dir, current_image + ".jpg")
                    cv2.imwrite(mark_img, image)

    
