# Before run this file, please follow Readme.md to download required model files.
# Usage example: python3 encapsulation_image.py --input greyscaleImage.png

import numpy as np
import cv2 as cv
import argparse
import os.path


def read_single_input():
    """
    :return: An image
    """
    parser = argparse.ArgumentParser(description='Colorize GreyScale Image')
    parser.add_argument('--input', help='Path to image.')
    args = parser.parse_args()

    if args.input == None:
        print('Please give the input greyscale image name.')
        print('Usage example: python3 colorizeImage.py --input greyscaleImage.png')
        exit()

    if os.path.isfile(args.input) == 0:
        print('Input file does not exist')
        exit()

    frame = cv.imread(args.input)

    return frame, args


def load_npy_file():
    """
    Load the 313 ab cluster centers from pts_in_hull.
    The color prediction task is a multinomial classification problem where for every gray pixel there are
    313 classes to choose from.
    :return: pts_in_hull
    """
    pts_in_hull = np.load('./pts_in_hull.npy')
    return pts_in_hull


def specify_and_read_model():
    """
    Read model files.
    Read the network into memory.
    :return:
    """
    protoFile = "./models/colorization_deploy_v2.prototxt"
    # Model with color rebalancing that contributes towards getting more vibrant and saturated colors in the output
    weightsFile = "./models/colorization_release_v2.caffemodel"
    # ⬇Model without color relalancing
    # weightsFile = "./models/colorization_release_v2_norebal.caffemodel"

    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    return net


def populate_cluster_centers(pts_in_hull, net):
    """
    Assign 1×1 kernels corresponding to each of the 313 bin centers and assign them to the corresponding layer
    in the network. And add a scaling layer with a non-zero value.
    :param pts_in_hull: Original readed pts_in_hull file
    :param net: Original readed network
    :return: the network
    """
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
    return net


def cvt_color_2_lab(frame):
    """
    Convert frame to LAB color space, for the further action of pulling out the L channel.
    :param frame: Original Frame
    :return: Frame in LAB color space
    """
    img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    return img_lab


def pull_out_L_channel(img_lab):
    """
    The grayscale image can be thought as the L-channel of the image in the LAB color space (lack of A/B channels).
    We take L-channel as the model input.
    :param img_lab: Frame in LAB color space
    :return: Lightness channel
    """
    img_l = img_lab[:, :, 0]
    return img_l


def resize_as_network_input_size(img_l, W_in, H_in):
    """
    Resize the lightness channel to the network input size
    :param img_l: Frame
    :param W_in, H_in: Network input size
    :return:
    """
    img_l_rs = cv.resize(img_l, (W_in, H_in))
    return img_l_rs


def mean_subtraction(img_l_rs):
    img_l_rs -= 50
    return img_l_rs


def run_caffe_net_forward(img_l_rs, net):
    """
    The training dataset of the model is tons of colored images and their corresponding grayscale images.
    The network predicts the a/b channels using the L channel.
    :return: Results
    """
    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # this is our result
    return ab_dec


def resize_as_original_size(frame, ab_dec):
    """
    Resize the output AB channels, which are predicted by the L channel, to the original frame size
    :param frame: Original frame
    :param ab_dec: Result
    :return: AB channels in original size
    """
    (H_orig, W_orig) = frame.shape[:2]  # original image size
    ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
    return ab_dec_us


def concatenate_channels(img_l, ab_dec_us):
    """
    This method concatenate the resized AB channels with the original L channel.
    :param img_l: Original L channel.
    :param ab_dec_us: AB channels in original size.
    :return: Colored image in original size.
    """
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)
    return img_bgr_out


def save_output(img_bgr_out, args):
    outputFile = args.input[:-4] + '_colorized.png'  # save
    # outputFile = args.input[:-4]+'_norebal_colorized.png'  # save
    cv.imwrite(outputFile, (img_bgr_out * 255).astype(np.uint8))
    print('Colorized image saved as ' + outputFile)
    print('Done !!!')

    
def main():
    # Import image
    frame, args = read_single_input()

    # Data preprocessing
    img_lab = cvt_color_2_lab(frame)
    img_l = pull_out_L_channel(img_lab)
    img_l_rs = resize_as_network_input_size(img_l, 224, 224)
    img_l_rs = mean_subtraction(img_l_rs)

    # Model and file import and setup
    pts_in_hull = load_npy_file()
    net = specify_and_read_model()
    net = populate_cluster_centers(pts_in_hull, net)

    # Run net forward
    ab_dec = run_caffe_net_forward(img_l_rs, net)

    # Scale the image back and merge three channels
    ab_dec_us = resize_as_original_size(frame, ab_dec)
    img_bgr_out = concatenate_channels(img_l, ab_dec_us)

    # Save image
    save_output(img_bgr_out, args)


if __name__ == "__main__":
    main()
