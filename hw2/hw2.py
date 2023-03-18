# import required module
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

FILE = 'climate_out'  # 'ftfm_out''news_out'
GROUND_FILE = 'climate_ground.txt'
# threshold = 0.9
min_shot_duration = 10 # in frames

def Hist(files):
    ## calculate histogram
    # bin_hist_list = []
    # for bin_unit in range(10): 
    hist_list = []
    for frame_path in files:
        frame = cv2.imread(frame_path)
        # Compute color histogram
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
        hist_list.append(hist)
    
        
    prec_list = []
    reca_list = []
    threshold_list = []
    for threshold_unit in range(1, 100):
        threshold = threshold_unit * 0.01
        threshold_list.append(threshold)
        ## find shot boundaries
        shot_boundaries = []
        for i in range(1, len(hist_list)):
            # Compute histogram difference
            diff = cv2.compareHist(hist_list[i-1], hist_list[i], cv2.HISTCMP_CORREL)
            if diff < threshold:
                shot_boundaries.append(i+1)
        
        ## count difference between shot lengths and 
        shot_lengths = np.diff(shot_boundaries)
        short_shots = np.where(shot_lengths < min_shot_duration)[0]
        
        for i in reversed(short_shots):
            if i == len(shot_boundaries)-1:
                shot_boundaries = shot_boundaries[:i+1]
            else:
                shot_boundaries = shot_boundaries[:i+1] + shot_boundaries[i+2:]
        
        ## get the score
        ground_file = open(GROUND_FILE, "r")
        ground_data = ground_file.read()
        ground_list = ground_data.split("\n")
        ground_file.close()
        ground_list = ground_list[4:]
        
        for i in range(len(ground_list)):
            if '~' in ground_list[i]:
                first, second = ground_list[i].split("~")
                ground_list[i] = (first, second)
                
                
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for shot in shot_boundaries:
            hit = 0
            for number in ground_list:
                if type(number) is tuple:
                    (first ,second ) = number
                    if int(first) <= shot <= int(second):
                        true_positive += 1
                        hit = 1
                        break
                else:
                    if shot == int(number):
                        true_positive += 1
                        hit = 1
                        break
                    
            if hit == 0:
                false_positive += 1
                
        for number in ground_list:
            hit = 0
            for shot in shot_boundaries:
                if type(number) is tuple:
                    (first ,second) = number
                    if int(first) <= shot <= int(second):
                        hit = 1
                        break
                else:
                    if shot == int(number):
                        hit = 1
                        break
            if hit == 0:
                false_negative += 1
                
        true_negative = len(files) - len(shot_boundaries) - false_negative
        try:
            precision = true_positive/(true_positive + false_positive)
        except:
            precision = 0
        try:
            recall = true_positive/(true_positive + false_negative)
        except:
            recall = 0
        print(precision, recall)
        prec_list.append(precision)
        reca_list.append(recall)
    return [prec_list, reca_list, threshold_list]

if __name__ == "__main__":
    # iterate over files in that directory
    files = [str(p) for p in Path(FILE).glob('*')]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    algorithm1_list = Hist(files)
    print(algorithm1_list)
    
    plt.style.use('seaborn-v0_8')
    # plot pre vs recall
    fig, ax = plt.subplots()
    ax.plot(algorithm1_list[0], algorithm1_list[1], linewidth=2.0)
    plt.xlabel('precision')
    plt.ylabel('recall')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    fig.set_size_inches(16.5, 9.5)
    fig.savefig(FILE + "_pre_recall" + '.png', dpi=100)
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))

    # plot pre + recall vs threshold granularity
    fig1, ax1 = plt.subplots()
    ax1.plot(algorithm1_list[2], algorithm1_list[0], label='precision', linewidth=2.0)
    ax1.plot(algorithm1_list[2], algorithm1_list[1], label='recall', linewidth=2.0)
    plt.xlabel('threshold')
    plt.ylabel('precision and recall')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.legend()
    fig1.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    fig1.set_size_inches(16.5, 9.5)
    fig1.savefig(FILE + "_threshold" + '.png', dpi=100)
    
    plt.show()
    
    