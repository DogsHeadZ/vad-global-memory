import torch
import yaml
import argparse
import torch.nn.functional as F

#
# class convAE(torch.nn.Module):
#
#     def __init__(self, n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
#         super(convAE, self).__init__()
#         print(n_channel,t_length, memory_size,feature_dim, key_dim,temp_update)
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--config')
#
# args = parser.parse_args()
# config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
#
# train_dataset_args = config['train_dataset_args']
#
# model = convAE(train_dataset_args['c'], train_dataset_args['t_length'], **config['model_args'])

#
#
# anomaly_score_total_list = []
# anomaly_score_total_list += [1,2]
# anomaly_score_total_list += [4,5]
# print(anomaly_score_total_list)
#
# items  = F.normalize(torch.rand((10, 64,3,3), dtype=torch.float),
#                 dim=1)
# print(items)
#
# print(int(2 / 1.2))

# import torch
# x = torch.FloatTensor([[1., 2.]])
# w1 = torch.FloatTensor([[2.], [1.]])
# w2 = torch.FloatTensor([3.])
# w1.requires_grad = True
# w2.requires_grad = True
#
# d = torch.matmul(x, w1)
# f = torch.matmul(d, w2)
# d = d.clone()
# d[:] = 1 # 因为这句, 代码报错了 RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#
# f.backward()

memorys = F.normalize(torch.rand((10, 10), dtype=torch.float),
                          dim=1).unsqueeze(0).repeat(64, 1, 1)  # Initialize the memory items
print(memorys)