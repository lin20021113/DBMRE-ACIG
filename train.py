"""
Training script for DMD
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from run import DMD_run
if __name__ == '__main__':
      DMD_run(model_name='dmd', dataset_name='mosi', is_tune=False, seeds=[1111,2222], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_distill=True)