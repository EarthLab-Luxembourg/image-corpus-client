"""
Setting Package that contains the URL to access the different services
in the Corpus API and in the Authentication API
"""

## FIXME: We need to check if we keep the staging...

class GlobalSetting:
    """
    Global Setting Object
    """
    IS_STAGING = False


class Settings(object):
    def __init__(self):
        """
        Setting Init
        """
        self.GET_CORPUS_URL = 'http://_STAGING_HOST_AND_PORT_/priv/training-corpus-api/v1/computer-vision-corpus/{corpus_id}/' \
            if GlobalSetting.IS_STAGING \
            else 'http://_PROD_HOST_AND_PORT_/priv/training-corpus-api/v1/computer-vision-corpus/{corpus_id}/'

        self.GET_ALL_IMAGES_URL = 'http://_STAGING_HOST_AND_PORT_/priv/training-corpus-api/v2/computer-vision-corpus/{corpus_id}/images/' \
            if GlobalSetting.IS_STAGING \
            else 'http://_PROD_HOST_AND_PORT_/priv/training-corpus-api/v2/computer-vision-corpus/{corpus_id}/images/'

        self.GET_IMAGE_BYTES = 'http://_STAGING_HOST_AND_PORT_/priv/training-corpus-api/v1/computer-vision-corpus/images/{image_id}/' \
            if GlobalSetting.IS_STAGING \
            else 'http://_PROD_HOST_AND_PORT_/priv/training-corpus-api/v1/computer-vision-corpus/images/{image_id}/'

        self.GET_IMAGE_ANNOTATION = 'http://_STAGING_HOST_AND_PORT_/priv/training-corpus-api/v1/computer-vision-corpus/images/{image_id}/annotations' \
            if GlobalSetting.IS_STAGING \
            else 'http://_PROD_HOST_AND_PORT_/priv/training-corpus-api/v1/computer-vision-corpus/images/{image_id}/annotations'

        self.JWT_TOKEN_URL_NEW = 'http://_STAGING_AUTH_HOST_AND_PORT_/authServer/api/v1/token/' \
            if GlobalSetting.IS_STAGING \
            else 'http://_PROD_AUTH_HOST_AND_PORT_/authServer/api/v1/token/'

        self.CREATE_CORPUS = 'http://_STAGING_AUTH_HOST_AND_PORT_/priv/training-corpus-api' \
            if GlobalSetting.IS_STAGING \
            else 'http://_PROD_AUTH_HOST_AND_PORT_/priv/training-corpus-api'
