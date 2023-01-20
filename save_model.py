import os

from mrcnn.config import Config
from mrcnn import model as modellib
from samples.shapes import shapes
from tensorflow import saved_model
from tensorflow.keras.backend import get_session

MODEL_DIR = "logs"
DEFAULT_WEIGHTS = os.path.join(MODEL_DIR, "mask_rcnn_shapes_heads.h5")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100
    STEPS_PER_EPOCH = 3
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(DEFAULT_WEIGHTS, by_name=True)

builder = saved_model.builder.SavedModelBuilder('saved_model/2023-01-18_shapes')
outs = model.keras_model.output

inputs = {}
for curr_input in model.keras_model.input:
        inputs[curr_input.name.split(':')[0].split('/')[0]] = curr_input

outputs = {}
for curr_output in model.keras_model.output:
        outputs[curr_output.name.split(':')[0].split('/')[0]] = curr_output

signature = saved_model.signature_def_utils.predict_signature_def(
            # dictionary of 'name' and model inputs (it can have more than one)
            inputs=inputs,
            # dictionary of 'name' and model outputs (it can have more than one)
            outputs=outputs)
            
signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
builder.add_meta_graph_and_variables(get_session(), [saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
builder.save()
