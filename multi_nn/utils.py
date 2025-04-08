# utils.py
import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import pandas as pd

class gc:  #graph characteristics
    def peaks(data):
        data=np.array(data)
        peaks=[]
        for i in range(1,len(data)-1):
            if data[i]>data[i-1] and data[i]>data[i+1]:
                peaks.append(i)
        return peaks
    def roots(data,tolerence,mean):
        data=np.array(data)
        roots=[]
        for i in range(1,len(data)-1):
            if abs(data[i]-mean)<tolerence:
                roots.append(i)
        return roots
    def valleys(data):
        data=np.array(data)
        valleys=[]
        for i in range(1,len(data)-1):
            if data[i]<data[i-1] and data[i]<data[i+1]:
                valleys.append(i)
        return valleys

class dnc:  #normal characteristics in deep learning
    def extreme_z_peaks(y_data):  # 3sigma peaks 
        y_data=np.array(y_data)
        return [idx for idx in gc.peaks(y_data) if y_data[idx]>np.mean(y_data[idx])+3*np.var(y_data)]
    def extreme_z_valleys(y_data): # 3sigma valleys
        y_data=np.array(y_data)
        return [idx for idx in gc.valleys(y_data) if y_data[idx]<np.mean(y_data[idx])-3*np.var(y_data)]
    def extreme_z_bsgm(y_data):
        segment=[]
        y_data=np.array(y_data)
        for peak in dnc.extreme_z_peaks(y_data):
            roots_left=max(z for z in gc.roots(y_data) if z<peak)
            roots_right=min(z for z in gc.roots(y_data) if z>peak)
            segment.append(y_data[roots_left:roots_right+1])
        return segment
    def extreme_z_btm(x_data):
        time=[]
        y_data=np.array(y_data)
        for peak in dnc.extreme_z_peaks(y_data):
            roots_left=max(z for z in gc.roots(y_data) if z<peak)
            roots_right=min(z for z in gc.roots(y_data) if z>peak)
            time.append(x_data[roots_left:roots_right+1])
        return time 
    def extreme_z_ssgm(y_data):
        segment=[]
        y_data=np.array(y_data)
        for valleys in dnc.extreme_z_valleys(y_data):
            roots_left=max(z for z in gc.roots(y_data) if z<valleys)
            roots_right=min(z for z in gc.roots(y_data) if z>valleys)
            segment.append(y_data[roots_left:roots_right+1])
        return segment
    def extreme_z_stm(x_data):
        time=[]
        x_data=np.array(x_data)
        for valleys in dnc.extreme_z_valleys(x_data):
            roots_left=max(z for z in gc.roots(x_data) if z<valleys)
            roots_right=min(z for z in gc.roots(x_data) if z>valleys)
            time.append(x_data[roots_left:roots_right+1])
        return time


class GCProcessor:
    """优化后的图形特征处理器"""
    
    @staticmethod
    def peaks(data):
        """向量化实现找波峰"""
        data = np.asarray(data)
        peaks, _ = find_peaks(data)
        return peaks.tolist()
    
    @staticmethod
    def valleys(data):
        """向量化实现找波谷"""
        data = np.asarray(data)
        valleys, _ = find_peaks(-data)
        return valleys.tolist()
    
    @staticmethod
    def roots(data, tolerance, mean):
        """向量化实现找零点"""
        data = np.asarray(data)
        mask = np.abs(data - mean) < tolerance
        roots = np.where(mask)[0]
        roots = roots[(roots > 0) & (roots < len(data)-1)]
        return roots.tolist()

class DataPreprocessor:
    """数据预处理类"""
    
    @staticmethod
    def load_and_process_data(file_path):
        """加载并预处理数据"""
        data = pd.read_csv(file_path)
        
        # 时间处理
        data['时间'] = pd.to_datetime(data['时间'], errors='coerce')
        start_time = data['时间'].min()
        data['时间'] = start_time + pd.to_timedelta(np.arange(len(data)) * 5, unit='ms')
        
        return data

    @staticmethod
    def process_wave(wave, is_extreme):
        """处理单个波形"""
        wave = np.asarray(wave)
        return np.ones(len(wave)) if is_extreme else np.zeros(len(wave))

    @staticmethod
    def prepare_training_data(data, tolerance=0.1):
        """准备训练数据"""
        y_data = data['加速度X(g)'].values
        mean = np.mean(y_data)
        
        # 找零点
        z_roots = GCProcessor.roots(y_data, tolerance, mean)
        
        # 分割波形
        all_waves = [y_data[z_roots[i]:z_roots[i+1]+1] 
                    for i in range(len(z_roots)-1)]
        
        # 判断极端条件
        is_extreme = dnc.extreme_z_peaks(y_data) or dnc.extreme_z_valleys(y_data)
        
        # 并行处理波形
        with ThreadPoolExecutor() as executor:
            labels = list(executor.map(
                partial(DataPreprocessor.process_wave, is_extreme=is_extreme),
                tqdm(all_waves, desc="分析波形")
            ))
        
        # 填充波形
        max_length = max(len(wave) for wave in all_waves)
        padded_waves = np.zeros((len(all_waves), max_length))
        for i, wave in enumerate(tqdm(all_waves, desc="填充波形")):
            padded_waves[i, :len(wave)] = wave
        
        return padded_waves, np.concatenate(labels)