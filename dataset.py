#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
import glob
import open3d  as o3d

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', 
            num_points=2048, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 
            'shapenetpart', 'modelnet10', 'modelnet40']
        assert num_points <= 2048        

        if dataset_name in ['shapenetpart', 'shapenetcorev2']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        self.root = os.path.join(root, dataset_name + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetpart', 'shapenetcorev2']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)    # load label name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_json_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.load_name:
            return point_set, label, name
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]


class MyDataset(data.Dataset):
    def __init__(self, root, npoints=8192, utransform=None):
        self.npoints = npoints
        self.root = root
        self.pointlist = []
        self.pointpath = root + "/point/"
        self.point_list = files = glob.glob(self.pointpath + "/*.txt")
        count=0
        for file in self.point_list:
            print(file)
            # point cloud 取得
            src = np.loadtxt(file)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(src)
            # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
            pcd = pcd.select_down_sample(ind)
            src = np.asarray(pcd.points)
            normlized_xyz = np.zeros((npoints, 3))
            self.coord_min, self.coord_max = np.amin(src, axis=0)[:3], np.amax(src, axis=0)[:3]
            # print(self.coord_min)
            # print(self.coord_max)
            if(self.coord_max[0] == 0):continue
            if(self.coord_max[1] == 0):continue
            if(self.coord_max[2] == 0):continue
            src[:, 0] = src[:, 0] - self.coord_min[0]
            src[:, 1] = src[:, 1] - self.coord_min[1]
            src[:, 2] = src[:, 2] - self.coord_min[2]
            if(len(src) >=npoints):
                np.random.shuffle(src)
                normlized_xyz[:,:]=src[:npoints,:]
            else:
                normlized_xyz[:len(src),:]=src[:,:]

            self.pointlist.append(normlized_xyz)
            # print(normlized_xyz.shape)
            count+=1
            # if(count==100):break
                

        self.data_num = len(self.pointlist)
        

    def __getitem__(self, index):
        point = self.pointlist[index]
        point = torch.from_numpy(point)
        return point

    def __len__(self):
        return len(self.pointlist)
