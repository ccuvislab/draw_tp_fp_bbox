import os

def get_tp_fp_fn_different(FedMA_file_path, FedAvg_file_path, tp_fp_diff_file_path):
    FedMA_data = {}
    with open(FedMA_file_path, 'r') as FedMA_file:
        for line in FedMA_file:
            parts = line.strip().split(': ')
            key = parts[0]
            values = parts[1].split(', ')
            tp = int(values[0].split('= ')[1])
            fp = int(values[1].split('= ')[1])
            FedMA_data[key] = {'tp': tp, 'fp': fp}

    FedAvg_data = {}
    with open(FedAvg_file_path, 'r') as FedAvg_file:
        for line in FedAvg_file:
            parts = line.strip().split(': ')
            key = parts[0]
            values = parts[1].split(', ')
            tp = int(values[0].split('= ')[1])
            fp = int(values[1].split('= ')[1])
            FedAvg_data[key] = {'tp': tp, 'fp': fp}

    with open(tp_fp_diff_file_path, 'w') as output_file:
        for key in FedAvg_data.keys():
            if key in FedMA_data:
                tp_diff = FedMA_data[key]['tp'] - FedAvg_data[key]['tp']
                fp_diff = FedMA_data[key]['fp'] - FedAvg_data[key]['fp']
                output_file.write(f"{key}: tp_diff = {tp_diff}, fp_diff = {fp_diff}\n")

def get_score_from_different(tp_fp_diff_file_path, output_score_file_path):
    scores = {}
    with open(tp_fp_diff_file_path, 'r') as diff_file:
        for line in diff_file:
            parts = line.strip().split(': ')
            key = parts[0]
            values = parts[1].split(', ')
            tp_diff = int(values[0].split('= ')[1])
            fp_diff = int(values[1].split('= ')[1])
            
            score = tp_diff + fp_diff
            scores[key] = score

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    with open(output_score_file_path, 'w') as output_file:
        for key, score in sorted_scores:
            output_file.write(f"{key}: score = {score}\n")

def main():
    FedMA_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp_fn/FedMA_tp_fp_fn_count.txt"
    FedAvg_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp_fn/FedAvg_tp_fp_fn_count.txt"
    tp_fp_fn_diff_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp_fn/tp_fp_diff.txt"
    output_score_file_path = "/home/u1755025/FedMPEN_mycode/output/tp_fp_fn/scores.txt"

    # different算法是FedMA - FedAvg
    get_tp_fp_fn_different(FedMA_file_path, FedAvg_file_path, tp_fp_fn_diff_file_path)
    get_score_from_different(tp_fp_fn_diff_file_path, output_score_file_path)


main()
