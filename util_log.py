import logging

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
