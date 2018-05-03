import pGan_fSplice
from maskgen.plugins import findPlugin
import os

"""JT code only used for plugin conversion"""
def transform(img, source, target, **kwargs):
	donor = kwargs['donor']
	pGan_fSplice.loadModel(os.path.join(findPlugin('GAN_face_splice'),pGan_fSplice.PREDICTOR_PATH))
	pGan_fSplice.splice_donor_recipient(source,donor,target)
	return None, None

def operation():
	return {
		'name': 'PasteSplice',
		'category': 'Paste',
		'software': 'GAN_face_splice',
		'version': '0.1',
		'arguments':{
			'donor':{
				'type': 'donor',
				'description': 'The donor image from which to select the face'
			}
		},
		'description':'Swap the faces of two photographs',
		'transitions':[
			'image.image'
		]
	}