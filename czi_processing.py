from pylibCZIrw import czi as pyczi
import json
import numpy as np
import os, sys


CHANNEL_AXIS = 1


def _metadata_size(metadata, axis_name, default=1):
    try:
        value = metadata['ImageDocument']['Metadata']['Information']['Image'][axis_name]
    except Exception:
        return default

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _scene_rectangles(czi_file):
    rectangles = czi_file.scenes_bounding_rectangle
    if rectangles:
        return rectangles

    rect = czi_file.total_bounding_rectangle
    if rect.w <= 0 or rect.h <= 0:
        return {}

    return {0: rect}


def load_czi(path_to_file):
	# scenes = {}
	# with pyczi.open_czi(path_to_file) as f:
	# 	metadata = f.metadata
	# 	n_channels = f.total_bounding_box['C'][1]  # → 4
    
	# 	for scene_idx, rect in f.scenes_bounding_rectangle.items():
	# 		channels = []
	# 		for c in range(n_channels):
	# 			img = f.read(
	# 				roi=(rect.x, rect.y, rect.w, rect.h),
	# 				plane={'C': c},
	# 				scene=scene_idx
	# 			)
	# 			channels.append(img)
			
	# 		# Stack along channel axis → shape (H, W, n_channels)
	# 		scenes[scene_idx] = np.concatenate(channels, axis=-1)

	# return scenes, metadata

    scenes = {}
    with pyczi.open_czi(path_to_file) as f:
        metadata = f.metadata
        channel_axis = CHANNEL_AXIS

        bbox = f.total_bounding_box
        c0, c1 = bbox.get("C", (0, 1))
        z0, z1 = bbox.get("Z", (0, 1))
        t0, t1 = bbox.get("T", (0, _metadata_size(metadata, "SizeT")))
        include_t_plane = "T" in bbox or (t1 - t0) > 1

        scene_rectangles = _scene_rectangles(f)
        if not scene_rectangles:
            raise ValueError("No readable scene or total image rectangle found in CZI")

        for scene_idx, rect in scene_rectangles.items():
            t_stack = []

            for t in range(t0, t1):
                c_stack = []

                for c in range(c0, c1):
                    z_stack = []

                    for z in range(z0, z1):
                        plane = {"C": c, "Z": z}
                        if include_t_plane:
                            plane["T"] = t

                        img = f.read(
                            roi=(rect.x, rect.y, rect.w, rect.h),
                            plane=plane,
                            scene=scene_idx,
                        )

                        # img is usually (Y, X, 1) for single-channel grayscale
                        img2d = np.squeeze(img, axis=-1)  # -> (Y, X)

                        z_stack.append(img2d)

                    # -> (Z, Y, X)
                    c_stack.append(np.stack(z_stack, axis=0))

                # -> (C, Z, Y, X)
                t_stack.append(np.stack(c_stack, axis=0))

            # -> (T, C, Z, Y, X)
            scenes[scene_idx] = np.stack(t_stack, axis=0)
    
    return scenes, metadata, channel_axis


def get_channel_names(metadata):
    channel_info = metadata['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']
    channel_names = []

    for idx, channel in enumerate(_as_list(channel_info)):
        if isinstance(channel, dict):
            channel_names.append(channel.get('@Name', f'Ch {idx}'))
        else:
            channel_names.append(f'Ch {idx}')

    return channel_names


# get the min and max across all the scenes
def get_intensity_extrema(scenes, channel_axis=CHANNEL_AXIS):
    
    sample = next(iter(scenes.values()))
    n_channels = sample.shape[channel_axis]

    minv = np.full(n_channels, np.inf)
    maxv = np.full(n_channels, -np.inf)

    for k, img in scenes.items():
        # Normalize negative axis
        ch_ax = channel_axis % img.ndim
        
        # All axes except the channel axis
        reduce_axes = tuple(ax for ax in range(img.ndim) if ax != ch_ax)

        scene_min = img.min(axis=reduce_axes)
        scene_max = img.max(axis=reduce_axes)

        minv = np.minimum(minv, scene_min)
        maxv = np.maximum(maxv, scene_max)

        # if print_running:
        #     print(f"Scene {k}, min = {scene_min}, max = {scene_max}")

    return minv, maxv


def _percentile_delta_one_channel(scenes, channel_idx, delta, nbins, channel_axis=CHANNEL_AXIS):
    hist = np.zeros(nbins, dtype=np.int64)

    for arr in scenes.values():
        ch_ax = channel_axis % arr.ndim
        channel_data = np.take(arr, channel_idx, axis=ch_ax)
        hist += np.bincount(channel_data.ravel(), minlength=nbins)

    cdf = np.cumsum(hist)
    total = cdf[-1]

    low_target = delta * total
    high_target = (1 - delta) * total

    low_val = np.searchsorted(cdf, low_target, side="left")
    high_val = np.searchsorted(cdf, high_target, side="left")

    return low_val, high_val


def percentile_delta_per_channel(scenes, delta=0.05, channel_axis=CHANNEL_AXIS):
    if not scenes:
        raise ValueError("scenes is empty")

    # Get first element without making a list
    first = next(iter(scenes.values()))

    if first.ndim < 4:
        raise ValueError(f"Expected (..., C, Z, Y, X), got {first.shape}")

    ch_ax = channel_axis % first.ndim

    if first.dtype == np.uint8:
        nbins = 256
    elif first.dtype == np.uint16:
        nbins = 65536
    else:
        raise TypeError(f"Expected uint8 or uint16, got {first.dtype}")

    n_channels = first.shape[ch_ax]

    # Validate all scenes
    for img in scenes.values():
        if img.ndim < 4:
            raise ValueError(f"Expected (..., C, Z, Y, X), got {img.shape}")
        if img.dtype != first.dtype:
            raise TypeError("All scenes must have same dtype")
        if img.shape[channel_axis % img.ndim] != n_channels:
            raise ValueError("All scenes must have same number of channels")

    lows = []
    highs = []

    for c in range(n_channels):
        low, high = _percentile_delta_one_channel(
            scenes, channel_idx=c, delta=delta, nbins=nbins, channel_axis=channel_axis
        )
        lows.append(low)
        highs.append(high)

    return np.asarray(lows), np.asarray(highs)


# have options on how to normalize: direct minmax or percentile with default of 1%ile
# this function will get the minv and maxv for the normalization for each channel
def get_intensity_stats_all_scenes(
    scenes,
    norm_mode='DirectMinMax',
    pctile=None,
    channel_axis=CHANNEL_AXIS,
):

    if norm_mode not in ['DirectMinMax', 'Percentile']:
        raise Exception('Normalization mode must be DirectMinMax or Percentile')
    elif norm_mode == 'DirectMinMax':
        minv, maxv = get_intensity_extrema(scenes, channel_axis)
    elif norm_mode == 'Percentile':
        if not pctile:
            raise Exception('Please enter a numeric percentile value')
        else:
            minv, maxv = percentile_delta_per_channel(scenes, pctile / 100, channel_axis)
    
    return minv, maxv



def convert_to_rgb_all_scenes(scenes, conversion_params):

    def convert_to_rgb_single_img(img_bgr, img_magenta):

        _indexer_red = [slice(None)] * img_ndim
        _indexer_red[channel_axis] = 2
        _indexer_red = tuple(_indexer_red)
        _indexer_blue = [slice(None)] * img_ndim
        _indexer_blue[channel_axis] = 0
        _indexer_blue = tuple(_indexer_blue)
        if convert_mode == 'RemoveFarRed':
            pass
        elif convert_mode == 'MergeRed': # add the 4th channel to the 3rd (R)
            img_bgr[_indexer_red] += img_magenta
        elif convert_mode == 'MergeMagenta': # add the 4th channel to the 1st (B) and 3rd
            img_bgr[_indexer_blue] += img_magenta
            img_bgr[_indexer_red] += img_magenta

        return img_bgr
    
    def normalize_single_image(img, minv, maxv):

        # the last dim of img must be the same as the len of minv and maxv
        minv = np.asarray(minv)
        maxv = np.asarray(maxv)
        shape = [1] * img.ndim
        if minv.ndim > 0:
            shape[channel_axis] = len(minv)

        minv = minv.reshape(shape)
        maxv = maxv.reshape(shape)

        return upper_intensity * (img - minv) / (maxv - minv)

    # input should be a dict with the keys convert_mode, norm_mode, pctile_value, norm_before_combine, norm_after_combine

    # the output is gonna be a new dict, with the same keys as scenes and the values being the converted image
    # img input is BGR

    img_type = next(iter(scenes.values())).dtype
    if img_type == np.uint8:
        upper_intensity = 2**8 - 1
        next_type = np.uint16
    elif img_type == np.uint16:
        upper_intensity = 2 ** 16 - 1
        next_type = np.uint32
    else:
        print('Image must be of type np.uint8 or np.uint16')
        return

    img_ndim = next(iter(scenes.values())).ndim
    channel_axis = conversion_params.get('channel_axis', CHANNEL_AXIS)
    n_channels = next(iter(scenes.values())).shape[channel_axis]
    modes = ['RemoveFarRed', 'MergeRed', 'MergeMagenta']
    convert_mode = conversion_params.get('convert_mode', 'RemoveFarRed')
    norm_mode = conversion_params.get('norm_mode', 'DirectMinMax')
    pctile_value = conversion_params.get('pctile_value', None)
    norm_before_combine = conversion_params.get('norm_before_combine', False)
    norm_after_combine = conversion_params.get('norm_after_combine', True)
    channel_assignment = conversion_params.get('channel_assignment', {'blue': 0, 'green': 1, 'red': 2, 'far_red': 3})

    if convert_mode not in modes:
        raise Exception(f'convert_mode must be one of {modes}')
    
    print(f'norm before merge?, {norm_before_combine}')
    
    minv, maxv = get_intensity_stats_all_scenes(
        scenes,
        norm_mode,
        pctile_value,
        channel_axis,
    ) # get two (n_channels,) arrays

    # scenes_out will be the 3 channel output dict, scenes_magenta is to compute the output
    scenes_out = {}
    scenes_magenta = {}
    channel_idcs = [channel_assignment['blue'], channel_assignment['green'], channel_assignment['red'], channel_assignment['far_red']]
    indexer_bgr = [slice(None)] * img_ndim
    indexer_bgr[channel_axis] = channel_idcs[:-1]
    indexer_bgr = tuple(indexer_bgr)
    indexer_magenta = [slice(None)] * img_ndim
    indexer_magenta[channel_axis] = channel_idcs[-1]
    indexer_magenta = tuple(indexer_magenta)

    print('channel_idcs: ', channel_idcs)
    print(indexer_bgr)

    print('converting the scenes')
    for k,img in scenes.items():
        print(f'scene {k}')
        if norm_before_combine:
            scenes_out[k] = normalize_single_image(
                img[indexer_bgr],
                minv[channel_idcs[:-1]],
                maxv[channel_idcs[:-1]],
            ).astype(next_type)
            scenes_magenta[k] = normalize_single_image(
                img[indexer_magenta],
                minv[channel_idcs[-1]],
                maxv[channel_idcs[-1]],
            ).astype(next_type)
        else:
            scenes_out[k] = img[indexer_bgr].astype(img_type)   
            scenes_magenta[k] = img[indexer_magenta].astype(img_type)

        scenes_out[k] = convert_to_rgb_single_img(scenes_out[k], scenes_magenta[k])
    
    if norm_after_combine:
        print('normalizing output scenes')
        nminv, nmaxv = get_intensity_stats_all_scenes(
            scenes_out,
            norm_mode,
            pctile_value,
            channel_axis,
        )
        for k, img in scenes_out.items():
            scenes_out[k] = normalize_single_image(img, nminv, nmaxv).clip(0,upper_intensity).astype(img_type)
    else:
        for k, img in scenes_out.items():
            scenes_out[k] = img.clip(0,upper_intensity).astype(img_type)

    print('Conversion complete')

    return scenes_out


###############################################
# DEPRECATED FUNCTIONS
###############################################

# function that gets, for one channel, a 1d array of pixels of all the scenes
def get_flattened_intensity_by_channel(scenes):

    n_channels = next(iter(scenes.values())).shape[-1]
    
    rows = []
    
    for c in range(n_channels):
        vals = [scene[..., c].ravel() for scene in scenes.values()]
        rows.append(np.concatenate(vals))
    
    return np.stack(rows, axis=0)

def normalize_single_channel(img):
    if len(img.shape) > 2:
        raise Exception('img must be 2D')
    maxv = np.max(img)
    minv = np.min(img)
    if maxv == minv:
        return np.zeros_like(img, dtype=np.uint16)
    return ((img - minv) / (maxv - minv) * 255).astype(np.uint16)

    
    # img_norm = np.copy(img)
    # if len(img.shape) == 3:
    #     print('normalizing 3 channels')
    #     for i in range(img.shape[-1]):
    #         img_norm[...,i] = normalize_single_channel(img[...,i])
    # elif len(img.shape) == 2:
    #     print('normalizing 1 channels')
    #     img_norm = normalize_single_channel(img)
    
    # return img_norm

def get_percentile_intensities(scenes, pctile=1):

    sorted_channel_intensities = get_flattened_intensity_by_channel(scenes)
    
    minv = np.percentile(sorted_channel_intensities, pctile, axis=1)
    maxv = np.percentile(sorted_channel_intensities, 100-pctile, axis=1)
    
    return minv, maxv

'''
# for a given 4-channel image, converts it to RGB with different modes
# def convert_to_rgb_by_scene(img, convertmode, normbeforecombine=False, normaftercombine=False):
    if img.shape[-1] != 4:
        raise Exception('input must have 4 channels')
    
    # img input is BGR
    modes = ['RemoveRed', 'MergeRed', 'MergeMagenta']
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

    if convertmode == 'RemoveRed':
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
'''
