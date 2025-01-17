### I put these functions temporarily here, to make the notebooks more readable.
### TO DO: It should be described and moved to modules

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import glob

from time import time, strftime, gmtime


#! Where is tf actually used?
#! seems like this is just superimposing the bb's and making a vid
class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        
        self.tf_version = int(tf.__version__.split('.')[0])
        
        if self.tf_version == 1:
            self.session = tf.Session()
            with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file_handle.read())
            
        else:
            self.session = tf.compat.v1.Session()
            
            with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
                graph_def = tf.compat.v1.GraphDef() 
                graph_def.ParseFromString(file_handle.read())
            
        tf.import_graph_def(graph_def, name="net")
        
        if self.tf_version == 1:
            self.input_var = tf.get_default_graph().get_tensor_by_name(
                "net/%s:0" % input_name)
            self.output_var = tf.get_default_graph().get_tensor_by_name(
                "net/%s:0" % output_name)
        else: # TF 2
            self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "%s:0" % input_name)
            self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)
                    
        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        self._run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

    
    def _run_in_batches(self, f, data_dict, out, batch_size):
        data_len = len(out)
        num_batches = int(data_len / batch_size)

        #! what is s or e
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
            out[s:e] = f(batch_data_dict)
        if e < len(out):
            batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
            out[e:] = f(batch_data_dict)
            
    
    
def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

        
def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image



def generate_video(fps, inputPath='./frames/', outputFile = './output_video.avi',
                   fourcc = 'DIVX',
                   length = None):
    """
    Generates a video based on given parametrers
    Arguments:
        fps        - frames per second
        inputPath  - folder with frames
        outputFile - video file
        fourcc     - string used to create an cv2.VideoWriter_fourcc object
        length     - Maximum lentgh of the video (in seconds). If none, all frames are used. 
    """
    t = time()
    out = None
    filenames = glob.glob(os.path.join(inputPath, '*.jpg'))
    filenames.sort()
    if length is None:
        nrfiles = len(filenames)
    else:
        nrfiles = int(min(len(filenames), fps * length))
    for i, filename in enumerate(filenames):
        if length is not None:
            if i > nrfiles:
                break
                
        if i%10 == 0:
            sys.stdout.write('{:.1f}% ({}/{}) done in {} seconds.          \r'.format(
                100*i/nrfiles, i, nrfiles,
                strftime('%M:%S', gmtime(time() - t))))
            
        img = cv2.imread(filename)
        if out is None:
            height, width, layers = img.shape
            size = (width,height)
            out = cv2.VideoWriter(outputFile,cv2.VideoWriter_fourcc(*fourcc), fps, size)
        out.write(img) 

    out.release()
    print ('100% done!                                                               ')