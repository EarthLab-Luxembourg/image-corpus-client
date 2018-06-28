import logging
from datetime import datetime

import requests
from requests import HTTPError

from ..settings import Settings

LOGGER = logging.getLogger(__name__)


class Auth(object):
    def __init__(self, user: str, password: str):
        self.user = user
        self.password = password
        self._jwt = None
        self._jwt_creation = None

    @property
    def jwt(self):
        if self._jwt and self._jwt_creation and (datetime.now() - self._jwt_creation).total_seconds() < 14 * 60:
            return self._jwt
        else:
            return self._get_new_jwt()

    def _get_new_jwt(self):
        payload = {
            "user_name": self.user,
            "password": self.password,
            "with_token": False
        }
        try:
            res = requests.post(Settings().JWT_TOKEN_URL_NEW, json=payload, timeout=10)
            res.raise_for_status()
        except HTTPError as e:
            LOGGER.error("request returned {}".format(e.response.status_code))
            LOGGER.error(e.response.text)
            raise e
        jwt = res.json()["access_token"]
        self._jwt = jwt
        self._jwt_creation = datetime.now()
        return jwt
