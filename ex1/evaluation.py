import numpy as np
import image_compressor


def get_reconstruction_error(pred_image, gt_image):
    err = np.sqrt(((gt_image.astype(np.float32) - pred_image.astype(np.float32))**2).mean())

    return err


# Get number of bytes for each datatype
def get_elem_size(elem_dtype):
    elem_dtype = str(elem_dtype)
    if elem_dtype == 'bool':
        return 1.0 / 8.0
    elif '8' in elem_dtype:
        return 1
    elif '16' in elem_dtype:
        return 2
    elif '32' in elem_dtype:
        return 4
    elif '64' in elem_dtype:
        return 8
    else:
        raise Exception


def get_size(im: np.array):
    # Return size of numpy array in bytes
    return im.size * get_elem_size(im.dtype)


def get_combined_score(reconstruction_error, compressed_im_size, codebook_size):
    weighted_compressed_im_size = compressed_im_size / 10.0
    weighted_codebook_size = codebook_size // 10**6

    return reconstruction_error + weighted_compressed_im_size + weighted_codebook_size


def compute_evaluation_score(compressed_images, groundtruths, codebook):
    assert len(compressed_images) == len(groundtruths)

    reconstructor = image_compressor.ImageReconstructor(codebook)

    total_err = 0.0
    total_size = 0.0
    for im_comp, gt in zip(compressed_images, groundtruths):
        im_recon = reconstructor.reconstruct(im_comp)  
        recon_err = get_reconstruction_error(im_recon, gt)
        total_err += recon_err

        total_size += (im_comp.size * get_elem_size(im_comp.dtype))
    
    codebook_size = codebook.size * get_elem_size(codebook.dtype)

    mean_err = total_err / len(groundtruths)
    mean_size = total_size / len(groundtruths)

    eval_score = get_combined_score(mean_err, mean_size, codebook_size)
    return eval_score, mean_err, mean_size, codebook_size