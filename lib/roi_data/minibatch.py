import numpy as np
import numpy.random as npr
import cv2

from core.config import cfg
import utils.blob as blob_utils


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data', 'rois', 'labels']
    return blob_names


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_image_blob(roidb)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    # print(im_blob.shape)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))

        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]

        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob
    blobs['im_info'] = np.array(
    [im_blob.shape[2], im_blob.shape[3], im_scales[0]],
    dtype=np.float32)

    return blobs, True, index, im_scales


def get_minibatch_tuple(roidb, extra_roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_blob_extra, im_scales = _get_image_blob_tuple(roidb, extra_roidb)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    blobs['data_extra'] = im_blob_extra
    # print(im_blob.shape)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    rois_blob_extra = np.zeros((0, 5), dtype=np.float32)
    labels_blob_extra = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)
        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))


        labels_extra, im_rois_extra = _sample_rois(extra_roidb[im_i], num_classes)
        # Add to RoIs blob
        rois_extra = _project_im_rois(im_rois_extra, im_scales[im_i])
        batch_ind_extra = im_i * np.ones((rois_extra.shape[0], 1))
        rois_blob_this_image_extra = np.hstack((batch_ind_extra, rois_extra))
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes_extra = np.round(rois_blob_this_image_extra * cfg.DEDUP_BOXES).dot(v)
            _, index_extra, inv_index_extra = np.unique(hashes_extra, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image_extra = rois_blob_this_image_extra[index_extra, :]

        rois_blob_extra = np.vstack((rois_blob_extra, rois_blob_this_image_extra))
        # Add to labels blob
        labels_blob_extra = np.vstack((labels_blob_extra, labels_extra))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob
    blobs['rois_extra'] = rois_blob_extra
    blobs['labels_extra'] = labels_blob_extra

    return blobs, True, index, im_scales

def get_multi_minibatch(roidb, num_classes, batchsize=2):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_multi_image_blob(roidb, batchsize)

    # assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    # print(im_blob.shape)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)
    num_rois = np.zeros(0)

    indexes = []
    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)
        assert im_i == 0
        # Add to RoIs blob
        for j in range(batchsize):
            rois = _project_im_rois(im_rois, im_scales[im_i][j])
            # batch_ind = j * np.ones((rois.shape[0], 1))
            batch_ind = np.zeros((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))

            if cfg.DEDUP_BOXES > 0:
                v = np.array([1, 1e3, 1e6, 1e9, 1e12])
                hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
                _, index, inv_index = np.unique(hashes, return_index=True,
                                                return_inverse=True)
                rois_blob_this_image = rois_blob_this_image[index, :]
                indexes.append(index)

            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
            num_rois = np.hstack((num_rois, len(rois_blob_this_image) * np.ones(1)))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob
    blobs['num_rois'] = num_rois
    return blobs, True, indexes, im_scales


def get_minibatch_bin(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    BIN_TYPE = cfg.OICR.MT.BIN_TYPE
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales, im_blob_bin, im_scales_bin = _get_image_blob_bin(roidb, BIN_TYPE)
    # im_blob, im_scales = _get_image_blob(roidb)

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    if BIN_TYPE == "flip":
        roidb_flip = flip_entry(roidb)

    blobs['data'] = im_blob
    blobs['data_bin'] = im_blob_bin
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    rois_bin_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois = _sample_rois(roidb[im_i], num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))

        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]

        rois_blob = np.vstack((rois_blob, rois_blob_this_image))


        ### bin img
        if BIN_TYPE == "flip":
            _, im_rois_bin = _sample_rois(roidb_flip[im_i], num_classes)
        else:
            im_rois_bin = im_rois
        rois_bin = _project_im_rois(im_rois_bin, im_scales_bin[im_i])
        rois_bin_blob_this_image = np.hstack((batch_ind, rois_bin))
        rois_bin_blob_this_image = rois_bin_blob_this_image[index, :]
        rois_bin_blob = np.vstack((rois_bin_blob, rois_bin_blob_this_image))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['rois_bin'] = rois_bin_blob
    blobs['labels'] = labels_blob

    return blobs, True, index, im_scales, im_scales_bin




def _sample_rois(roidb, num_classes):
    """Generate a random sample of RoIs"""
    labels = roidb['gt_classes']
    rois = roidb['boxes']

    if cfg.TRAIN.BATCH_SIZE_PER_IM > 0:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_IM
    else:
        batch_size = np.inf
    if batch_size < rois.shape[0]:
        rois_inds = npr.permutation(rois.shape[0])[:batch_size]
        rois = rois[rois_inds, :]

    return labels.reshape(1, -1), rois


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_image_blob_tuple(roidb, extra_roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    processed_ims_extra = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

        im_extra = cv2.imread(extra_roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(extra_roidb[i]['image'])
        if extra_roidb[i]['flipped']:
            im_extra = im_extra[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im_extra, im_scale_extra = blob_utils.prep_im_for_blob(
            im_extra, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        processed_ims_extra.append(im_extra[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    blob_extra = blob_utils.im_list_to_blob(processed_ims_extra)

    return blob, blob_extra, im_scales


def _get_multi_image_blob(roidb, size=2):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=(num_images, size))
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        im_scales_per_image = []
        # processed_ims_per_image = []
        for j in range(size):
            target_size = cfg.TRAIN.SCALES[scale_inds[i][j]]
            im_chg, im_scale = blob_utils.prep_im_for_blob(
                im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
            im_scales_per_image.append(im_scale[0])
            processed_ims.append(im_chg[0])
        im_scales.append(im_scales_per_image)
        # processed_ims.append(processed_ims_per_image)
    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def _get_image_blob_bin(roidb, BIN_TYPE):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    if BIN_TYPE == "scale":
        scale_inds_bin = np.random.randint(
            0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    processed_ims_bin = []
    im_scales_bin = []

    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        im_ori = im.copy()
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

        ### flip
        if BIN_TYPE == "flip":
            im_b = im_ori[:, ::-1, :]
        else:
            im_b = im_ori
        if BIN_TYPE == "scale":
            target_size_b = cfg.TRAIN.SCALES[scale_inds_bin[i]]
        else:
            target_size_b = target_size

        im_b, im_scale_b = blob_utils.prep_im_for_blob(
            im_b, cfg.PIXEL_MEANS, [target_size_b], cfg.TRAIN.MAX_SIZE)
        im_scales_bin.append(im_scale_b[0])
        processed_ims_bin.append(im_b[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    blob_bin = blob_utils.im_list_to_blob(processed_ims_bin)

    return blob, im_scales, blob_bin, im_scales_bin


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def flip_entry(roidb):
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    return flipped_roidb