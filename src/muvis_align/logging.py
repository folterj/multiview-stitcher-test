import logging
import os
from  multiview_stitcher import __version__ as mvs_version

from muvis_align._version import version


def init_logging(log_filename='log/muvis-align.log', log_format='%(asctime)s %(levelname)s: %(message)s', verbose=False):
    basepath = os.path.dirname(log_filename)
    if basepath and not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    handlers = [logging.FileHandler(log_filename, encoding='utf-8')]
    if verbose:
        handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, encoding='utf-8', force=True)

    # verbose external modules
    if verbose:
        # expose multiview_stitcher.registration logger and make more verbose
        mvsr_logger = logging.getLogger('multiview_stitcher.registration')
        mvsr_logger.setLevel(logging.DEBUG)
        if len(mvsr_logger.handlers) == 0:
            mvsr_logger.addHandler(logging.StreamHandler())
    else:
        # reduce verbose level
        for module in ['multiview_stitcher', 'multiview_stitcher.registration', 'multiview_stitcher.fusion']:
            logging.getLogger(module).setLevel(logging.WARNING)

    for module in ['ome_zarr']:
        logging.getLogger(module).setLevel(logging.WARNING)

    logging.info(f'muvis-align version {version}')
    logging.info(f'Multiview-stitcher version: {mvs_version}')
