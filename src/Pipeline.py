import logging
import os
from tqdm import tqdm

from src.image.source_helper import get_images_metadata
from src.util import dir_regex, get_filetitle, find_all_numbers, split_underscore_numeric


class Pipeline:
    def __init__(self, params):
        self.params = params
        self.params_general = params['general']
        self.init_logging()
        from src.MVSRegistration import MVSRegistration
        self.mvs_registration = MVSRegistration(self.params_general)

    def init_logging(self):
        verbose = self.params_general.get('verbose', False)
        verbose_mvs = self.params_general.get('verbose_mvs', False)
        log_filename = self.params_general.get('log_filename', 'logfile.log')
        log_format = self.params_general.get('log_format')
        basepath = os.path.dirname(log_filename)
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        handlers = [logging.FileHandler(log_filename, encoding='utf-8')]
        if verbose:
            handlers += [logging.StreamHandler()]

        logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, encoding='utf-8')

        # verbose external modules
        if verbose_mvs:
            # expose multiview_stitcher.registration logger and make more verbose
            mvsr_logger = logging.getLogger('multiview_stitcher.registration')
            mvsr_logger.setLevel(logging.INFO)
            if len(mvsr_logger.handlers) == 0:
                mvsr_logger.addHandler(logging.StreamHandler())
        else:
            # reduce verbose level
            for module in ['multiview_stitcher', 'multiview_stitcher.registration', 'multiview_stitcher.fusion']:
                logging.getLogger(module).setLevel(logging.WARNING)

        for module in ['ome_zarr']:
            logging.getLogger(module).setLevel(logging.WARNING)

    def run(self):
        break_on_error = self.params_general.get('break_on_error', False)

        for operation_params in tqdm(self.params['operations']):
            input_path = operation_params['input']
            logging.info(f'Input: {input_path}')
            try:
                self.run_operation(operation_params)
            except Exception as e:
                logging.exception(f'Error processing: {input_path}')
                print(f'Error processing: {input_path}: {e}')
                if break_on_error:
                    break

        logging.info('Done!')

    def run_operation(self, params):
        operation = params['operation']
        filenames = dir_regex(params['input'])
        verbose = self.params_general.get('verbose', False)
        if len(filenames) == 0:
            logging.warning(f'Skipping operation {operation} (no files)')
            return

        operation_parts = operation.split()
        if 'match' in operation_parts:
            # sort last key first
            filenames = sorted(filenames, key=lambda file: list(reversed(find_all_numbers(get_filetitle(file)))))
            if len(operation_parts) > operation_parts.index('match') + 1:
                match_label = operation_parts[-1]
            else:
                match_label = 's'
            matches = {}
            for filename in filenames:
                parts = split_underscore_numeric(filename)
                match_value = parts.get(match_label)
                if match_value is not None:
                    if match_value not in matches:
                        matches[match_value] = []
                    matches[match_value].append(filename)
                if len(matches) == 0:
                    matches[0] = filenames
            filesets = list(matches.values())
            fileset_labels = [match_label + label for label in matches.keys()]
            logging.info(f'# matched file sets: {len(filesets)}')
        else:
            filesets = [filenames]
            fileset_labels = ['']

        if verbose:
            for fileset, fileset_label in zip(filesets, fileset_labels):
                logging.info(f'File set: {fileset_label} metadata:\n' + get_images_metadata(fileset))

        for fileset, fileset_label in zip(filesets, fileset_labels):
            if len(filesets) > 1:
                logging.info(f'File set: {fileset_label}')
            self.mvs_registration.run_operation(fileset, params)
