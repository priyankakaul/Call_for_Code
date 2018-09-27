# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IBM_API_KEY_ID': 'iWKNJ0MSajfQIdOLAeuw_LPfHxMz9Cp-8RmBYwCatNHo',
    'IAM_SERVICE_ID': 'iam-ServiceId-1ab8d756-eec2-4e5a-b87f-4523082d2867',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
    'BUCKET': 'agro-donotdelete-pr-aczjmpgkn5brjd',
    'FILE': '7.jpg'
}

cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=credentials_1['IBM_API_KEY_ID'],
    ibm_service_instance_id=credentials_1['IAM_SERVICE_ID'],
    ibm_auth_endpoint=credentials_1['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=credentials_1['ENDPOINT'])
	

cos.download_file(Bucket=credentials_1['BUCKET'],Key='7.jpg',Filename='7.jpg')

import json
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
	'2018-03-19',
	url='https://gateway.watsonplatform.net/visual-recognition/api',
	iam_apikey='V9a7ynzCeC1sKh1HQJkefZqDBNKsynfL69lD3x7EwLmy')
	

	with open('7.jpg', 'rb') as images_file:
	classes = visual_recognition.classify(
		images_file,
		threshold='0.64',
		classifier_ids='DefaultCustomModel_1322275209')
	print(json.dumps(classes, indent=2))