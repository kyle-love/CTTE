import re
import numpy as np
import torch
import json
import random

def get_ids(file_path):
    fin = open(file_path).readlines()
    id_list = list(map(lambda x: re.split(r'[: ]', x)[0], fin))
    return set(id_list)

def save_data(path, data, seperator='\t'):
    data.sort(key=lambda x: x[0])
    out = open(path, 'w')
    for line in data:
        # 转化为字符串格式
        line = list(map(lambda x: str(x), line))
        out.write(seperator.join(line) + '\n')
    out.close()

def get_list_data(data, seperator_pattern= r'[:_ |]'):
    return list(map(lambda x: re.split(seperator_pattern, x.strip()), data))

def get_public_ids(datasets):
    # 获取共同的ids
    public_ids = None

    for dataset_title in datasets:
        dataset_data = datasets[dataset_title]
        ids_set = set(map(lambda x:x[0], dataset_data))
        print('Unique Id Count in {0} Data: {1}'.format(dataset_title, len(ids_set)))
        print(ids_set.pop())
        if public_ids == None:
            public_ids = ids_set
        else:
            public_ids = public_ids & ids_set

    print('Public Unique Id Count in Datasets: {0}'.format(len(public_ids)))
    return public_ids

def extract_public_data(data, public_ids):
    new_data = []
    for line in data:
        if line[0] in public_ids:
            new_data.append(line)
    return new_data

def filter_by_public_ids(list_data, public_ids):
    return list(filter(lambda x: x[0] in public_ids, list_data))

def merge_continious_data_by_id(speed_datas):
    speed_data = []
    nums = len(speed_datas)
    rows = len(speed_datas[0])
    for i in range(rows):
        id = None
        new_line = None
        for j in range(nums):
            if id is None:
                id = speed_datas[j][i][0]
                new_line = [id]
            else:
                if id != speed_datas[j][i][0]:
                    print('Warning: Id Mismatch!')
            new_line.extend(speed_datas[j][i][1:])
        speed_data.append(new_line)
    return speed_data
    

def process_dataset():
    speed_file_names = [
        "link_speed_0813",
        "link_speed_0814",
        "link_speed_0815",
        "link_speed_0816",
        "link_speed_0817"
        ]
    query_file_names = [
        "link_query_0813",
        "link_query_0814",
        "link_query_0815",
        "link_query_0816",
        "link_query_0817"
    ]

    speed_dir = "dataset/raw/link_speed/"
    query_dir = "dataset/raw/link_query/"
    embedings_dir = "dataset/raw/embedings/"

    #求共有的ids

    public_ids = None

    #获取速度的共有ids
    public_speed_ids = None

    for speed_file_name in speed_file_names:
        if public_speed_ids is None:
            public_speed_ids = get_ids(speed_dir + speed_file_name)
        else:
            public_speed_ids = public_speed_ids & get_ids(speed_dir + speed_file_name)
    print('Count of Public Speed Ids: {0}'.format(len(public_speed_ids)))
    
    #获取查询的共有ids
    public_query_ids = None

    for query_file_name in query_file_names:
        if public_query_ids is None:
            public_query_ids = get_ids(query_dir + query_file_name)
        else:
            public_query_ids = public_query_ids & get_ids(query_dir + query_file_name)
    print('Count of Public Query Ids: {0}'.format(len(public_query_ids)))

    #获取Embedings的ids

    embedings_ids = None

    embedings_file_name = "8dim.embedings"
    embedings_ids = get_ids(embedings_dir + embedings_file_name)
    print('Count of Embedings Ids: {0}'.format(len(embedings_ids)))


    public_ids = public_speed_ids & public_query_ids & embedings_ids
    print('Count of Public Ids: {0}'.format(len(public_ids)))


    #获取的共有的ids之后，需要对数据进行处理，并且统一以\t分隔符进行存储

    #处理连续的数据，根据共有的ids筛选，并且将其根据ID合并
    speed_datas = []
    for speed_file_name in speed_file_names:
        list_data = get_list_data(open(speed_dir + speed_file_name).readlines())
        list_data = filter_by_public_ids(list_data, public_ids)
        list_data.sort(key=lambda x: x[0], reverse=True)
        speed_datas.append(list_data)
    speed_data = merge_continious_data_by_id(speed_datas)
    save_data('dataset/process/link_speed/link_speed_embedings', speed_data)
    
    start_query_datas = []
    end_query_datas = []
    for query_file_name in query_file_names:
        list_data = get_list_data(open(query_dir + query_file_name).readlines())
        list_data = filter_by_public_ids(list_data, public_ids)
        list_data.sort(key=lambda x: x[0], reverse=True)
        start_query_datas.append(list(map(lambda x: x[:289], list_data)))

        end_query_data = []
        for line in list_data:
            end_query_line = []
            end_query_line.append(line[0])
            end_query_line.extend(line[289:])
            end_query_data.append(end_query_line)
        end_query_datas.append(end_query_data)
    
    start_query_data = merge_continious_data_by_id(start_query_datas)
    save_data('dataset/process/link_query/start_query_embedings', start_query_data)

    end_query_data = merge_continious_data_by_id(end_query_datas)
    save_data('dataset/process/link_query/end_query_embedings', end_query_data)

    # 提取有效的embedings数据

    link_embedings = open(embedings_dir + embedings_file_name).readlines()
    list_data = get_list_data(link_embedings)
    list_data = filter_by_public_ids(list_data, public_ids)
    list_data.sort(key=lambda x: x[0], reverse=True)
    save_data('dataset/process/embedings/link_embedings', list_data)


def load_from_file(path, delimiter='\t'):
    data = np.loadtxt(path, delimiter=delimiter)
    data = data[:,1:]
    return data

def generate_dataset(time_step=12):
    speed_data = load_from_file('dataset/process/link_speed/link_speed_embedings')
    start_query_data = load_from_file('dataset/process/link_query/start_query_embedings')
    end_query_data = load_from_file('dataset/process/link_query/end_query_embedings')
    embedings_data = load_from_file('dataset/process/embedings/link_embedings')
    time_offset_line = [i for i in range(288)] * 7
    time_offset_data = np.asarray([time_offset_line for _ in range(speed_data.shape[0])])

    print("Speed Data Shape: " + str(speed_data.shape))
    print("Start Query Data Shape: " + str(start_query_data.shape))
    print("End Query Data Shape: " + str(end_query_data.shape))
    print("Embedings Data Shape: " + str(embedings_data.shape))

    # 每time_step个5分钟预测下一个5分钟
    features = []
    labels = []

    for i in range(288, len(speed_data[0]) - time_step - 1):
        new_data = None
        for j in range(time_step):
            new_time_offset_data = time_offset_data[:, [i+j]]
            new_speed_data = speed_data[:, [i+j]]
            new_start_query_data = start_query_data[:, [i+j]]
            new_end_query_data = end_query_data[:, [i+j]]
            if new_data is None:
                new_data = np.hstack((embedings_data[:], new_time_offset_data, new_speed_data, new_start_query_data, new_end_query_data))[np.newaxis, :, :]
            else:
                new_data = np.vstack((new_data, np.hstack((embedings_data[:], new_time_offset_data, new_speed_data, new_start_query_data, new_end_query_data))[np.newaxis, :, :]))

        period_data = np.hstack((embedings_data[:], time_offset_data[:, [i+j - 288]], speed_data[:, [i+j - 288]], start_query_data[:, [i+j - 288]], end_query_data[:, [i+j - 288]]))
        period_data = period_data[np.newaxis, :, :]

        new_data = np.vstack((new_data, period_data))

        features.append(new_data[np.newaxis, :, :, :])
        new_target = speed_data[:, [i+time_step]][np.newaxis, :, :]
        labels.append(new_target[np.newaxis, :, :, :])
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    np.save("dataset/processed/features.npy", features)
    np.save("dataset/processed/labels.npy", labels)

def process_mm1():
    #mmgps = 'original_mmgps_0817'
    mmgps = ['original_mmgps_0813','original_mmgps_0814','original_mmgps_0815','original_mmgps_0816','original_mmgps_0817']
    #mmgps_out = 'mmgps_0817'
    mmgps_out = ["mmgps_0813", "mmgps_0814", "mmgps_0815", "mmgps_0816", "mmgps_0817"]
    mmgps_dir = 'dataset/raw/mm/'
    mmgps_out_dir = 'dataset/process/mm/'
    for f in range(len(mmgps)):
        test = open(mmgps_dir + mmgps[f]).readlines()
        fout = open(mmgps_out_dir + mmgps_out[f], 'w')
    #test = open(mmgps_dir + mmgps).readlines()
    #fout = open(mmgps_out_dir + mmgps_out, 'w')
        for line in test:
            line = line.split(':')
            # 去除driver_id
            # driver_id = line[0]
            data = line[1].strip()[2:-3].split('),(')
            for traj in data:
                traj_data = traj.split(',')[1]
                traj_data = traj_data[:-1].split('|') # 去除末尾多余的 |
                traj_data = list(map(lambda x: '_'.join(x.split('_')[:-1]), traj_data))
                fout.write('\t'.join(traj_data) + '\n')
        fout.close()

def format(line, window_size=50):
    #what's the meaning of window size

    line = list(map(lambda x: x.split('_'), line))
    
    trajs = []
    
    length = len(line)

    if length < 2:
        return trajs

    if length < window_size:
        traj = []
        for _ in range(window_size - length):
            traj.append(['0', '0', '0', '0'])
        traj.extend(line)
        trajs.append(traj)
        return trajs

    for i in range(length):
        traj = []
        if i > 0:
            if i == length - 1:
                temp = []
                for _ in range(window_size - (i % window_size)):
                    temp.append(['0', '0', '0', '0'])
                traj.extend(line[int(i/window_size):i])
                continue
            if i % window_size == 0:
                traj.extend(line[i-window_size:i])
            else:
                continue
        else:
            continue
        trajs.append(traj)
    return trajs

def process_mm2(window_size=200):

    mm = ["mmgps_0813", "mmgps_0814", "mmgps_0815", "mmgps_0816", "mmgps_0817"]
    mm_dir = "dataset/process/mm/"
    mm_data = []
    for mm_file_name in mm:
        mm_data.extend(open(mm_dir + mm_file_name).readlines())
    mm_data = list(map(lambda x: x.strip().split('\t'), mm_data))
    mm_data.sort(key=lambda x: len(x))
    formatted_traj_data = []
    for line in mm_data:
        result = format(line, window_size)
        formatted_traj_data.extend(result)
    json.dump(formatted_traj_data, open('dataset/process/mm/formatted_mm', 'w'))

def process_mm3(window_size=50):
    data = json.load(open('dataset/process/mm/formatted_mm'))
    count = 0
    for line in data:
        if len(line) != window_size:
            count += 1
    print(count)

def process_mm4():
    public_ids = set()
    data = json.load(open('dataset/process/mm/formatted_mm'))
    for line in data:
        for item in line:
            link_id = item[3]
            if link_id != '0':
                public_ids.add(link_id)
    json.dump(list(public_ids), open('dataset/process/mm/public_ids', 'w'))

def process_mm5():
    public_ids = json.load(open('dataset/process/mm/public_ids'))
    embedings = open('dataset/raw/embedings/8dim.embedings').readlines()
    data = {}
    for line in embedings:
        line = line.strip().split(' ')
        if line[0] in public_ids:
            data[line[0]] = line[1:]
    json.dump(data, open('dataset/process/mm/mm_embedings', 'w'))

def process_mm6():
    mm_embedings =  json.load(open('dataset/process/mm/mm_embedings'))
    data = json.load(open('dataset/process/mm/formatted_mm'))
    new_data = []
    invalid = False
    for line in data:
        new_line = []
        for item in line:
            if item[-1] == '0':
                new_item = item[:-1]
                new_item.extend(['0', '0', '0', '0', '0', '0', '0', '0'])
            else:
                new_item = item[:-1]
                if item[-1] in mm_embedings:
                    new_item.extend(mm_embedings[item[-1]])
                else:
                    invalid = True
                    break
            new_line.append(new_item)
        if invalid:
            invalid = False
            continue
        new_data.append(new_line)
    json.dump(new_data, open('dataset/processed/traj_200.json', 'w'))

def process_mm7():
    data = json.load(open('dataset/processed/traj_200.json'))
    data = list(map(lambda x: list(map(lambda x: list(map(lambda x: float(x), x)), x)), data))
    json.dump(data, open('dataset/processed/traj_200', 'w'))

def generate_label():
    data = json.load(open('dataset/processed/traj_samples_10000'))
    data = list(map(lambda x: list(map(lambda x: list(map(lambda x: float(x), x)), x)), data))
    times = []
    for line in data:
        for item in line:
            if item[0] != 0:
                time = line[-1][0] - item[0]
                times.append(time)
                break
    print(len(times))
    json.dump(data, open('dataset/processed/traj_samples_10000_float', 'w'))
    json.dump(times, open('dataset/processed/time_samples_10000', 'w'))

def clear_time():
    data = json.load(open('dataset/processed/traj_samples_10000_float'))
    new_data = []
    for line in data:
        new_line = []
        for item in line:
            new_line.append(item[1:])
        new_data.append(new_line)
    print(len(new_data[0]))
    json.dump(new_data, open('dataset/processed/traj_samples_10000_without_time', 'w'))

def get_ordered_random_slices(data_list):
    if len(data_list) == 2:
        return data_list
    new_data_list = []

    indexes = []
    candidate = [i for i in range(len(data_list))]
    if len(candidate) > 4:
        #random sample
        indexes.extend(random.sample(candidate, int(len(candidate) * 0.8)))
        indexes.sort()
    else:
        indexes.extend(candidate)
    #what's the meaning
    if int(len(indexes) * 0.5) > 4:
        indexes = indexes[:int(len(indexes) * 0.1 * random.randint(5, 10))]

    for i in indexes:
        new_data_list.append(data_list[i])
    return new_data_list

def random_cut():
    traj_samples = json.load(open('dataset/processed/traj_200.json'))
    new_traj_samples = []
    count = 0
    for traj_sample in traj_samples:
        # 计算轨迹数量
        origin_length = len(traj_sample)
        new_traj_sample = []
        valid_points = []
        for point in traj_sample:
            if point[0] != '0':
                valid_points.append(point)
        valid_points = get_ordered_random_slices(valid_points)
        for _ in range(origin_length - len(valid_points)):
            # why append the same zero
            new_traj_sample.append(['0','0','0','0','0','0','0','0','0','0','0'])

        for point in valid_points:
            new_traj_sample.append(point)
        new_traj_samples.append(new_traj_sample)
        count = count + 1
        print(count)
    random.shuffle(new_traj_samples)
    json.dump(new_traj_samples, open('dataset/processed/traj_samples', 'w'))
    json.dump(new_traj_samples[:100010], open('dataset/processed/traj_samples_10000', 'w'))
    
if __name__ == '__main__':
    random_cut()
    generate_label()
    clear_time()