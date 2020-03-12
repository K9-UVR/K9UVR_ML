from absl import app, flags, logging
import os
import glob
import tensorflow as tf

""" Script for converting model frozen model to .tflite format. """

"""
Use here cli version example:

tflite_convert
--graph_def_file=/tmp/tflite/tflite_graph.pb
--output_file=detect.tflite
--input_shapes=1,320,320,3
--input_arrays=normalized_input_image_tensor
--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3
--allow_custom_ops
"""

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "path to trained model (.pb file)")
flags.DEFINE_string("output_name", None,
                    "path to output name of .tflite file.")

# Required flags
flags.mark_flag_as_required("model")
flags.mark_flag_as_required("output_name")


def conv(output_tensor):
    '''
    graph_def_file: Full filepath of file containing frozen GraphDef.
    input_arrays: List of input tensors to freeze graph with.
    output_arrays: List of output tensors to freeze graph with.
    input_shapes: Dict of strings representing input tensor names to list of
    integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
    Automatically determined when input shapes is None (e.g., {"foo" : None}). (default None)
    '''

    graph_def_file = FLAGS.model
    input_arrays = ["normalized_input_image_tensor"]
    output_arrays = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
                     'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3']
    input_shapes = {"normalized_input_image_tensor": [1, 320, 320, 3]}

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file,
        input_arrays,
        output_arrays,
        input_shapes
    )
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open("./converted_models/"+FLAGS.output_name, "wb").write(tflite_model)


def main(argv):
    del argv  # Unused
    conv(output_tensor)


if __name__ == '__main__':
    app.run(main)
