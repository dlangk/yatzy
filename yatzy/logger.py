import logging

logging.Formatter(fmt='%(asctime)s.%(msecs)03d', datefmt='%Y-%m-%d,%H:%M:%S')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(filename)-15s %(message)s',
    level=logging.INFO)


class YatzyLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def get_logger(self):
        return self.logger
