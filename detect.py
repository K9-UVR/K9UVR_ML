from absl import app, flags, logging
import tensorflow_hub as hub
import utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# urllib.request.urlopen(urllink)


FLAGS = flags.FLAGS
# flags.DEFINE_integer("num_times", 1, "Number of times to print greeting.")
# flags.DEFINE_string('name', None, 'Your name.')
# flags.DEFINE_integer('age', 25, 'Your age in years.', lower_bound=0)
# flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')

# Required flag.
# flags.mark_flag_as_required("name")


def main(argv):
	logging.set_verbosity(logging.INFO)
	del argv  # Unused.
#   for i in range(0, FLAGS.num_times):
#     print('Hello, %s!' % FLAGS.name)

	# image_url = "https://img-9gag-fun.9cache.com/photo/a6N211m_460swp.webp"
	image_url = "https://img-9gag-fun.9cache.com/photo/aWEbGgZ_460swp.webp"
	downloaded_image_path = utils.download_and_resize_image(
		image_url, 1280, 856, False) # false to not display empty img

	'''	Available models for model detection:
		"https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
		"https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
	'''
	# module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
	module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
	detector = hub.load(module_handle).signatures['default']
	# utils.load_img(downloaded_image_path)
	utils.run_detector(detector, downloaded_image_path)


if __name__ == '__main__':
	app.run(main)
