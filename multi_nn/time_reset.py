import pandas as pd
def time_reset(data_list):
    for data in data_list:   
        data['时间'] = pd.to_datetime(data['时间'],errors='coerce')
        start_time = data['时间'].min()
        data['时间'] = [start_time + pd.Timedelta(milliseconds=5 * i) for i in range(len(data))]   
    return data_list 