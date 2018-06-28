"""
Package to implement the Request Wrapper
"""

import logging

import requests
from requests import HTTPError, Response
from requests.compat import quote
from retrying import retry

from ..auth import Auth

LOGGER = logging.getLogger(__name__)


@retry(wait_exponential_multiplier=1000, stop_max_delay=60)
def get(url: str, auth: Auth, **kwargs) -> Response:
"""Wrapper to the request GET to handle log and retry

:param url: URL to call
:type url: str
:param auth: Auth Object to use when requesting URL
:type auth: Auth
:raises err: HTTPError
:return: Request Response Object
:rtype: Response
"""
    url = quote(url, safe='/:')
    try:
        res = requests.get(url, headers={'Authorization': 'Bearer {}'.format(auth.jwt), 'accept': 'application/json'}, timeout=20, **kwargs)
        res.raise_for_status()
    except HTTPError as err:
        LOGGER.error("request returned {}".format(err.response.status_code))
        LOGGER.error(err.request.url, err.response.text)
        raise err
    return res


@retry(wait_exponential_multiplier=1000, stop_max_delay=60)
def post(url: str, payload: dict, auth: Auth) -> Response:
"""Wrapper to the request POST to handle log and retry

:param url: URL to call
:type url: str
:param auth: Auth Object to use when requesting URL
:type auth: Auth
:param payload: Post Payload to send
:type payload: dict
:raises err: HTTPError
:return: Request Response Object
:rtype: Response
"""
    url = quote(url, safe='/:')
    try:
        res = requests.post(url, json=payload, headers={'Authorization': 'Bearer {}'.format(auth.jwt)}, timeout=20)
        res.raise_for_status()
    except HTTPError as err:
        LOGGER.error("request returned {}".format(err.response.status_code))
        LOGGER.error(err.request.url, err.response.text)
        raise err
    return res
