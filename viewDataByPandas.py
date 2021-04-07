import pandas as pd
import numpy as np

# filepath = './dataset/process/embedings/link_embedings'
# filepath = './dataset/process/link_query/end_query_embedings'
# filepath = './dataset/process/link_query/start_query_embedings'
# filepath = './dataset/process/link_speed/link_speed_embedings'
# filepath = './dataset/process/mm/formatted_mm'
# filepath = './dataset/process/mm/mm_embedings'
# filepath = './dataset/process/mm/mmgps_0813'


# filepath = './dataset/raw/embedings/8dim.embedings'
# filepath = './dataset/raw/link_query/link_query_0813'
# filepath = './dataset/raw/link_speed/link_speed_0813'
# filepath = './dataset/raw/mm/original_mmgps_0813'
# filepath = './dataset/raw/unclassified/link_speed(0813-0819)/link_speed_0813'
# filepath = './dataset/raw/unclassified/query(0813-0819)/link_query_0813'
# filepath = './dataset/raw/unclassified/0813-0819北京mm/original_mmgps_0813'



# data = pd.read_csv(filepath)

data = pd.read_csv(filepath, delim_whitespace=True)

# raw 中的数据
# raw 中的数据 可以用 delim_whitespace = True参数分割
# data = pd.read_csv(filepath, delimiter='_')
# 288+288 = 576
# raw中的 link_query数据可以通过_分割 288个时间片 * 2 （起始、终点）
# raw中的 link_speedy数据可以通过_分割 288个时间片
# raw 中 unclassified 列数差不多

# process中数据
# link_embedings 用的八维                                                    1907行
# end_query_embedings start_query_embedings 列数都是1441=288*5 每分钟？？ 都是1907行
# link_speed_embedings                      列数都是1441=288*5 每分钟？？ 都是1907行
#
#

print(data.shape[1])
