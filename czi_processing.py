from pylibCZIrw import czi as pyczi
import json
import numpy as np
import os, sys


def load_czi(path_to_file):
    
	scenes = {}
	with pyczi.open_czi(path_to_file) as f:
		metadata = f.metadata
		n_channels = f.total_bounding_box['C'][1]  # → 4
    
		for scene_idx, rect in f.scenes_bounding_rectangle.items():
			channels = []
			for c in range(n_channels):
				img = f.read(
					roi=(rect.x, rect.y, rect.w, rect.h),
					plane={'C': c},
					scene=scene_idx
				)
				channels.append(img)
			
			# Stack along channel axis → shape (H, W, n_channels)
			scenes[scene_idx] = np.concatenate(channels, axis=-1)

	return scenes, metadata


def get_channel_names(metadata):
     
    channel_names = []
    channel_info = metadata['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']
    for d in channel_info:
        channel_names.append(d['@Name'])
    
    return channel_names   


def normalize(img):

    def normalize_single(img):
        if len(img.shape) > 2:
            raise Exception('img must be 2D')
        maxv = np.max(img)
        minv = np.min(img)
        if maxv == minv:
            return np.zeros_like(img, dtype=np.uint16)
        return ((img - minv) / (maxv - minv) * 255).astype(np.uint16)

    
    img_norm = np.copy(img)
    if len(img.shape) == 3:
        print('normalizing 3 channels')
        for i in range(img.shape[-1]):
            img_norm[...,i] = normalize_single(img[...,i])
    elif len(img.shape) == 2:
        print('normalizing 1 channels')
        img_norm = normalize_single(img)
    
    return img_norm


# for a given 4-channel image, converts it to RGB with different modes
def convert_to_rgb(img, convertmode, normbeforecombine=False, normaftercombine=False):
    if img.shape[-1] != 4:
        raise Exception('input must have 4 channels')
    
    # img input is BGR
    modes = ['Remove647', 'MergeRed', 'MergeMagenta']
    print(f"convertmode: {convertmode}")
    if convertmode not in modes:
        raise Exception(f'convertmode must be one of {modes}')

    # img_out and img_far flip the channel axis to RGB
    if normbeforecombine:
        img_out = normalize(img[...,-2::-1]).astype(np.uint16)
        img_far = normalize(img[...,-1]).astype(np.uint16)
        print('img was normalized prior to rgb conversion')
    else:
        img_out = img[...,-2::-1].astype(np.uint16)   
        img_far = img[...,-1].astype(np.uint16)
        print('img was NOT normalized prior to rgb conversion')

    if convertmode == 'Remove647':
        pass
    elif convertmode == 'MergeRed': # add the 4th channel to the 1st (R)
        img_out[...,0] += img_far
    elif convertmode == 'MergeMagenta': # add the 4th channel to the 1st and 3rd (B)
        img_out[...,0] += img_far
        img_out[...,2] += img_far

    if normaftercombine:
        return normalize(img_out).clip(0,255).astype(np.uint8)
    else:
        return img_out.clip(0,255).astype(np.uint8)