import pGan_fSplice
from maskgen.plugins import findPlugin
import os

"""JT code only used for plugin conversion"""


def transform(img, source, target, **kwargs):
	donor = kwargs['donor']
	pGan_fSplice.loadModel(os.path.join(findPlugin('GAN_face_splice'), pGan_fSplice.PREDICTOR_PATH))
	pGan_fSplice.splice_donor_recipient(source, donor, target)
	return {'subject': 'face', 'purpose': 'add'}, None


def operation():
	return {
		'name': 'PasteSplice',
		'category': 'Paste',
		'software': 'GAN_face_splice',
		'version': '0.1',
		'arguments': {
			'donor': {
				'type': 'donor',
				'description': 'The donor image from which to select the face'
			},
			'donor rotated': {
				'type': 'yesno',
				'description': 'Enter yes if the donor is rotated during the paste operation',
				'default value': 'yes'
			},
			'donor cropped': {
				'type': 'yesno',
				'description': 'Enter yes if the donor is cropped during the paste operation. '
							   'Ideally, crop should occur as a SelectRegion operation just prior to donation',
				'default value': 'no'
			},
			'donor resized': {
				'type': 'yesno',
				'description': 'Enter yes if the donor is resized during the paste operation',
				'default value': 'yes'
			},
			'purpose': {
				'type': 'list',
				'values': [
					'remove',
					'add',
					'blend'
				],
				'description': 'Purpose: remove an object, add an object.',
				'default value': 'add'
			},
			'subject': {
				'type': 'list',
				'values': [
					'people',
					'face',
					'natural object',
					'man-made object',
					'large man-made object',
					'landscape',
					'other'
				],
				'default value': 'face'
			}
		},
		'description': 'Swap the faces of two photographs',
		'transitions': [
			'image.image'
		]
	}
