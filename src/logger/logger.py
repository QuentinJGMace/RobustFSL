import os
import logging


class Logger:
    def __init__(self, module_name, filename):
        self.module_name = module_name
        self.filename = filename
        self.formatter = self.get_formatter()
        self.file_handler = self.get_file_handler()
        self.stream_handler = self.get_stream_handler()
        self.logger = self.get_logger()

    def get_formatter(self):
        log_format = "[%(name)s]: [%(levelname)s]: %(message)s"
        formatter = logging.Formatter(log_format)
        return formatter

    def get_file_handler(self):
        file_handler = logging.FileHandler(self.filename)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        return stream_handler

    def get_logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.file_handler)
        logger.addHandler(self.stream_handler)
        return logger

    def del_logger(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def exception(self, msg):
        self.logger.exception(msg)


def make_log_dir(log_path, dataset, method):
    log_dir = os.path.join(log_path, dataset, method)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_log_file(log_path, dataset, method):
    log_dir = make_log_dir(log_path=log_path, dataset=dataset, method=method)
    i = 0
    filename = os.path.join(log_dir, "{}_run_{}.log".format(method, i))
    while os.path.exists(os.path.join(log_dir, "{}_run_%s.log".format(method)) % i):
        i += 1
        filename = os.path.join(log_dir, "{}_run_{}.log".format(method, i))
    return filename
