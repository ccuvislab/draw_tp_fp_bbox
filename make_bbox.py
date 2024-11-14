import os
import cv2


def drawFP_TP_BB(FedMA_file_path, top_10_img,img_path,output_image_dir):
    #針對FedMA或FedAvg，做標框動作
    with open(FedMA_file_path, "r") as file:
        current_image = None
        for line in file:
            #若bb.txt中存在top 10照片名稱(基本一定存在)，分別對該照片，取出tp及fp bbox
            if any(image_name in line for image_name in top_10_img):
                current_image = line.strip().split(": ")[0]
                
                tp_content = eval(line.split("tp = ")[1].split(", fp = ")[0].strip())
                fp_content = eval(line.split("fp = ")[1].strip())

                image_files = os.listdir(img_path)
                #從特地路徑抓取top 10照片來框取
                if current_image + ".jpg" in image_files:
                    image_path = os.path.join(img_path, current_image + ".jpg")
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
                        # fp_count = 0
                        for fp_bb in fp_content:
                            #使用前面提取的fp bbox
                            x_min, y_min, x_max, y_max = map(int, fp_bb)
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                            # fp_count += 1
                            # if fp_count > 3:
                            #     break
                
                        #框好照片儲存在FedMA或FedAvg的file底下
                        #output_image_dir = "output/mark_FedMA_bb"
                        if not os.path.exists(output_image_dir):
                            os.makedirs(output_image_dir)
                        mark_img = os.path.join(output_image_dir, current_image + ".jpg")
                        cv2.imwrite(mark_img, image)

    

def get_top_10_img(output_score_file_path):

    draw_list = []
    with open(output_score_file_path, "r") as file:
        image_score = []
        for line in file:
            parts = line.split(": score = ")
            img_name = parts[0].strip()
            score = int(parts[1].strip())
            
            image_score.append((img_name, score))

    #依照score由高到低排列
    image_score.sort(key = lambda x: x[1], reverse = True)
    draw_list = [image_score[i][0] for i in range(min(10, len(image_score)))]
    return draw_list


def get_specific_condition(FedAvg_file_path):
    #read from FedAvg_file_path if TP>10 and FP<10, save to draw_list
    #format is frankfurt_000001_029600_leftImg8bit: tp = 5, fp = 17

    draw_list = []
    with open(FedAvg_file_path, "r") as file:
        current_image = None
        for line in file:
            #若bb.txt中存在top 10照片名稱(基本一定存在)，分別對該照片，取出tp及fp bbox
            current_image = line.strip().split(": ")[0]
            tp_content = eval(line.split("tp = ")[1].split(", fp = ")[0].strip())
            fp_content = eval(line.split("fp = ")[1].strip())
           # print(tp_content)

            if len(tp_content) > 10 and len(fp_content) < 20:
                draw_list.append(current_image)
    return draw_list

def main():

    img_path = "/home/superorange5/Research/ProbabilisticTeacher/data/VOC2007_citytrain/JPEGImages"
    output_score_file_path = "input_data/scores.txt"
    FedMA_file_path = "input_data/FedMA_tp_fp_bb.txt"
    FedAvg_file_path = "input_data/FedAvg_tp_fp_bb.txt"
    output_folder_MA = "output/mark_FedMA_bb"
    output_folder_Avg = "output/mark_FedAvg_bb"
    output_folder = 'output/FedAvg_test2'

    
    #draw_list = get_top_10_img(output_score_file_path)
    draw_list = get_specific_condition(FedAvg_file_path)  # draw by tp count 
    # draw_list = ['frankfurt_000000_001236_leftImg8bit']
    

    

    #draw top 10
    drawFP_TP_BB(FedAvg_file_path,draw_list,img_path,output_folder)


if __name__ == "__main__":
    main()