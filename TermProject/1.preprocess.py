import os,shutil,tqdm
import cv2
import numpy as np

current_path = os.path.dirname(__file__)
#traindata_path = f'{current_path}/train_data'
traindata_path = f'{current_path}/train'

for root, dirs, files in os.walk(traindata_path):
    
    if files != []:     #當有走到存在圖片的資料夾才取圖
        root_b = root.split('scans')[0]

        try:
            shutil.rmtree(f'{root_b}/normalize')
        except:
            pass
        os.makedirs(f'{root_b}/normalize')
        

        for img_name in tqdm.tqdm(files):
            # 讀取16位元PNG圖像
            img = cv2.imread(f'{root}/{img_name}', cv2.IMREAD_UNCHANGED)
            '''
            # 設定窗位和窗寬
            window_center = 50
            window_width = 350
            
            # 調整窗位和窗寬
            min_value = window_center - window_width // 2
            max_value = window_center + window_width // 2
            img = np.clip(img, min_value, max_value)
            # 線性轉換為0-255的範圍
            img = ((img - min_value) / window_width * 255).astype('uint8')
            '''
            img = (img - img.min())/(img.max() - img.min())*255.0

            # 儲存為PNG格式的可視化圖像
            cv2.imwrite(f'{root_b}/normalize/{img_name}', img)
        try:
            shutil.rmtree(f'{root}')
        except:
            pass