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


# get the min and max across all the scenes
def get_intensity_extrema(scenes, print_running=False):
    
    n_channels = next(iter(scenes.values())).shape[-1]

    minv = np.full(n_channels, np.inf)
    maxv = np.full(n_channels, -np.inf)

    for k, img in scenes.items():
        scene_min = img.min(axis=(0, 1))
        scene_max = img.max(axis=(0, 1))

        minv = np.minimum(minv, scene_min)
        maxv = np.maximum(maxv, scene_max)

        if print_running:
            print(f"Scene {k}, min = {scene_min}, max = {scene_max}")

    return minv, maxv

# function that gets, for one channel, a 1d array of pixels of all the scenes
def get_flattened_intensity_by_channel(scenes):

    n_channels = next(iter(scenes.values())).shape[-1]
    
    rows = []
    
    for c in range(n_channels):
        vals = [scene[..., c].ravel() for scene in scenes.values()]
        rows.append(np.concatenate(vals))
    
    return np.stack(rows, axis=0)


def get_percentile_intensities(scenes, pctile=1):

    sorted_channel_intensities = get_flattened_intensity_by_channel(scenes)
    
    minv = np.percentile(sorted_channel_intensities, pctile, axis=1)
    maxv = np.percentile(sorted_channel_intensities, 100-pctile, axis=1)
    
    return minv, maxv


# have options on how to normalize: direct minmax or percentile with default of 1%ile
# this function will get the minv and maxv for the normalization for each channel
def get_intensity_stats_all_scenes(scenes, norm_mode='DirectMinMax', pctile=None):

    if norm_mode not in ['DirectMinMax', 'Percentile']:
        raise Exception('Normalization mode must be DirectMinMax or Percentile')
    elif norm_mode == 'DirectMinMax':
        minv, maxv = get_intensity_extrema(scenes)
    elif norm_mode == 'Percentile':
        if not pctile:
            raise Exception('Please enter a numeric percentile value')
        else:
            minv, maxv = get_percentile_intensities(scenes, pctile)
    
    return minv, maxv

def normalize_single_image(img, minv, maxv):

    # the last dim of img must be the same as the len of minv and maxv

    return 255 * (img - minv) / (maxv - minv)

def convert_to_rgb_all_scenes(scenes, conversion_params):

    def convert_to_rgb_single_img(img_bgr, img_magenta):

        if convert_mode == 'Remove647':
            pass
        elif convert_mode == 'MergeRed': # add the 4th channel to the 3rd (R)
            img_bgr[...,2] += img_magenta
        elif convert_mode == 'MergeMagenta': # add the 4th channel to the 1st (B) and 3rd
            img_bgr[...,0] += img_magenta
            img_bgr[...,2] += img_magenta

        return img_bgr
    
    # input should be a dict with the keys convert_mode, norm_mode, pctile_value, norm_before_combine, norm_after_combine

    # the output is gonna be a new dict, with the same keys as scenes and the values being the converted image
    # img input is BGR
    n_channels = next(iter(scenes.values())).shape[-1]
    modes = ['Remove647', 'MergeRed', 'MergeMagenta']
    # print(f"convert_mode: {convert_mode}")
    convert_mode = conversion_params.get('convert_mode', 'Remove647')
    norm_mode = conversion_params.get('norm_mode', 'DirectMinMax')
    pctile_value = conversion_params.get('pctile_value', None)
    norm_before_combine = conversion_params.get('norm_before_combine', False)
    norm_after_combine = conversion_params.get('norm_after_combine', True)

    if convert_mode not in modes:
        raise Exception(f'convert_mode must be one of {modes}')
    
    print(f'norm before merge?, {norm_before_combine}')
    
    minv, maxv = get_intensity_stats_all_scenes(scenes, norm_mode, pctile_value) # get two (n_channels,) arrays

    # scenes_out will be the 3 channel output dict, scenes_magenta is to compute the output
    scenes_out = {}
    scenes_magenta = {}
    print('converting the scenes')
    for k,img in scenes.items():
        print(f'scene {k}')
        if norm_before_combine:
            scenes_out[k] = normalize_single_image(img[...,:n_channels-1], minv[:n_channels-1], maxv[n_channels-1]).astype(np.uint16)
            scenes_magenta[k] = normalize_single_image(img[...,-1], minv[:-1], maxv[-1]).astype(np.uint16)
        else:
            scenes_out[k] = img[...,:n_channels-1].astype(np.uint16)   
            scenes_magenta[k] = img[...,-1].astype(np.uint16)

        scenes_out[k] = convert_to_rgb_single_img(scenes_out[k], scenes_magenta[k])
    
    if norm_after_combine:
        print('normalizing output scenes')
        nminv, nmaxv = get_intensity_stats_all_scenes(scenes_out, norm_mode, pctile_value)
        for k, img in scenes_out.items():
            scenes_out[k] = normalize_single_image(img, nminv, nmaxv).clip(0,255).astype(np.uint8)
    else:
        for k, img in scenes_out.items():
            scenes_out[k] = img.clip(0,255).astype(np.uint8)

    return scenes_out

# rewrite this function to take minv and maxv as arguments, both would be (n_channels,) arrays
def normalize_single_image_old(img):

    def normalize_single_channel(img):
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
            img_norm[...,i] = normalize_single_channel(img[...,i])
    elif len(img.shape) == 2:
        print('normalizing 1 channels')
        img_norm = normalize_single_channel(img)
    
    return img_norm


# for a given 4-channel image, converts it to RGB with different modes
def convert_to_rgb_by_scene(img, convertmode, normbeforecombine=False, normaftercombine=False):
    if img.shape[-1] != 4:
        raise Exception('input must have 4 channels')
    
    # img input is BGR
    modes = ['Remove647', 'MergeRed', 'MergeMagenta']
    print(f"convertmode: {convertmode}")
    if convertmode not in modes:
        raise Exception(f'convertmode must be one of {modes}')

    # img_out and img_far flip the channel axis to RGB
    if normbeforecombine:
        img_out = normalize_single_image_old(img[...,-2::-1]).astype(np.uint16)
        img_far = normalize_single_image_old(img[...,-1]).astype(np.uint16)
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
        return normalize_single_image_old(img_out).clip(0,255).astype(np.uint8)
    else:
        return img_out.clip(0,255).astype(np.uint8)
    

# def get_intensity_array_all_scenes(scenes):