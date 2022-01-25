import cv2
import torch
import numpy as np

import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import glob
import copy
from random import randint, choice
import json
from PIL import Image

from torchvision.datasets.vision import VisionDataset

def load_video(video_fname):
    frames = []
    cap = cv2.VideoCapture(video_fname)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
    #print(len(frames), video_fname)
    return frames

class TemporalSample(object):
    def __init__(self, clip_len=16, stride=1, random=True):
        self.clip_len = clip_len
        self.stride = stride
        self.random = random
        
    def __call__(self,sample):
        flow_len = len(sample)
        start =0
        if self.random:
            max_index = flow_len - self.stride * self.clip_len
            #print(flow_len, max_index)
            start = np.random.randint(0,flow_len - self.stride * self.clip_len)
            try:
                start = np.random.randint(0, max_index)
            except ValueError:
                raise ValueError
        else:
            start = flow_len // 2
        end = start + self.clip_len * self.stride
        sample = sample[start:end:self.stride]
        sample = np.stack(sample)
        return sample


def load_img_seq(video_fname, seq_list):
    ####
    img_seq = []
    for i in seq_list:
        img_i = f'{i:06d}' + '.jpg'
        video_frame_path = os.path.join(video_fname, img_i)
        video_frame = np.array(Image.open(video_frame_path).convert("RGB"))
        img_seq.append(video_frame)
    sample = np.stack(img_seq)
    return sample

class VideoFlowFramesDataset(VisionDataset):
    def __init__(self, folder='kinetics_frames', contrastive=True, train=True, transform=None, clip_len=16, stride=1):
        self.folder = folder
        
        self.all_videos_with_flow = glob.glob(os.path.join(self.folder,'**/*/*'), recursive=False)
        f = open(os.path.join(self.folder, 'classids.json'))
        self.class_ids = json.load(f)
        f.close()
        self.transform = transform
        key = 'val_256'
        if train:
            key = 'train_256' 
        self.all_videos_with_flow = [f for f in self.all_videos_with_flow if key in f]#[:96]
        print(f'{len(self.all_videos_with_flow)} videos found')
        self.train = train
        self.contrastive=contrastive
        self.clip_len = clip_len
        self.stride = stride
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.train:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def __len__(self):
        return len(self.all_videos_with_flow)
    
    def __getitem__(self, index):
        #flow_fname = self.all_videos_with_flow[index]
        video_fname= self.all_videos_with_flow[index]#.replace(self.flow_folder,self.video_folder)
        #print(flow_fname, video_fname)
        #flow = load_video(flow_fname) # list of 3ch images
        #rgb = load_img_seq(os.path.join(rgb)) # list of 3ch images
        classid = self.class_ids[os.path.basename(os.path.dirname(video_fname))]
        #rgb_time_crop = self.temporal_sample(rgb)
        rgb_path = os.path.join(video_fname, 'rgb')
        flow_path = os.path.join(video_fname, 'flow')
        try:
            flow_len = len(os.listdir(flow_path))
            start = 0
            if self.train:
                max_index = flow_len - self.stride * self.clip_len
                #print(flow_len, max_index)
                #start = np.random.randint(0,flow_len - self.stride * self.clip_len)
                #start = np.random.randint(0, max_index)
                if ((max_index // self.stride)  <= self.clip_len) or (max_index <= 0):
                    #print((max_index - start) <= self.clip_len, max_index)
                    #print("skipping sample 1")
                    return self.skip_sample(index)
                else:
                    start = np.random.randint(0, max_index)
            else:
                start = flow_len // 2
            end = start + self.clip_len * self.stride
            rgb_time_crop = load_img_seq(rgb_path, list(range(start + 1, end + 1, self.stride)))
            rgb_time_crop = torch.from_numpy(rgb_time_crop)
            if self.contrastive:
                sample1 = self.transform(rgb_time_crop)
                sample2 = self.transform(rgb_time_crop)
                sample = (sample1, sample2)
            else:
                sample = self.transform(rgb_time_crop)    
        except:
            return self.skip_sample(index)
            #print("skipping sample 2")
        return sample, classid
    
class VideoDataset(VisionDataset):
    def __init__(self, video_folder='official', contrastive=True, train=True, transform=None, clip_len=16, stride=1):
        self.video_folder = video_folder
        
        self.all_videos = glob.glob(os.path.join(video_folder,'**/*.mp4'), recursive=True)
        f = open(os.path.join(video_folder, 'classids.json'))
        self.class_ids = json.load(f)
        f.close()
        self.transform = transform
        key = 'val_256'
        if train:
            key = 'train_256' 
        self.all_videos = [f for f in self.all_videos if key in f][:128]
        print(f'{len(self.all_videos)} videos found')
        self.shuffle = train
        self.contrastive=contrastive
        self.clip_len=clip_len
        self.stride=stride
        self.temporal_sample = TemporalSample(clip_len=self.clip_len, stride=self.stride, random=train)
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def __len__(self):
        return len(self.all_videos)
    
    def __getitem__(self, index):
        #flow_fname = self.all_videos_with_flow[index]
        video_fname= self.all_videos[index]
        #print(flow_fname, video_fname)
        #flow = load_video(flow_fname) # list of 3ch images
        rgb = load_video(video_fname) # list of 3ch images
        classid = self.class_ids[os.path.basename(os.path.dirname(video_fname))]
        #rgb_time_crop = self.temporal_sample(rgb)
        try:
            rgb_time_crop = torch.from_numpy(self.temporal_sample(rgb))
            if self.contrastive:
                sample1 = self.transform(rgb_time_crop)
                sample2 = self.transform(rgb_time_crop)
                sample = (sample1, sample2)
            else:
                sample = self.transform(rgb_time_crop)
        except:
            return self.skip_sample(index)
        return sample, classid

class VideoFlowDataset(VisionDataset):
    def __init__(self, video_folder='official', flow_folder='official_flow', contrastive=True, train=True, transform=None):
        self.video_folder = video_folder
        self.flow_folder = flow_folder
        
        self.all_videos_with_flow = glob.glob(os.path.join(flow_folder,'**/*.mp4'), recursive=True)
        f = open(os.path.join(video_folder, 'classids.json'))
        self.class_ids = json.load(f)
        f.close()
        self.transform = transform
        key = 'val_256'
        if train:
            key = 'train_256' 
        self.all_videos_with_flow = [f for f in self.all_videos_with_flow if key in f]#[:96]
        print(f'{len(self.all_videos_with_flow)} videos found')
        self.shuffle = train
        self.contrastive=contrastive
        self.temporal_sample = TemporalSample()
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def __len__(self):
        return len(self.all_videos_with_flow)
    
    def __getitem__(self, index):
        #flow_fname = self.all_videos_with_flow[index]
        video_fname= self.all_videos_with_flow[index].replace(self.flow_folder,self.video_folder)
        #print(flow_fname, video_fname)
        #flow = load_video(flow_fname) # list of 3ch images
        rgb = load_video(video_fname) # list of 3ch images
        classid = self.class_ids[os.path.basename(os.path.dirname(video_fname))]
        #rgb_time_crop = self.temporal_sample(rgb)
        try:
            rgb_time_crop = torch.from_numpy(self.temporal_sample(rgb))
            if self.contrastive:
                sample1 = self.transform(rgb_time_crop)
                sample2 = self.transform(rgb_time_crop)
                sample = (sample1, sample2)
            else:
                sample = self.transform(rgb_time_crop)
        except:
            return self.skip_sample(index)
        return sample, classid
    
    
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import os
import torch

class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.
    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.
    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.
    Internally, it uses a VideoClips object to handle clip creation.
    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('avi',), transform=None, num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0):
        super(Kinetics400, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)#[1:513]
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            None,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        self.transform = transform
    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        # video_q, audio_q, info_q, video_idx_q = self.video_clips.get_clip(idx[0])
        # video_k, audio_k, info_k, video_idx_k = self.video_clips.get_clip(idx[1])
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        video_q = self.transform(video)
        video_k = self.transform(video)
#         audio_q = self.transform['audio'](audio)
#         audio_k = self.transform['audio'](audio)

        return (video_q, video_k), label#, (audio_q, audio_k)