from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
import pandas as pd
from threading import Thread
from tqdm import tqdm

from muvis_align.image.source_helper import get_images_metadata
from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.logging import init_logging
from muvis_align.util import dir_regex, get_filetitle, find_all_numbers, find_target_numeric


class Pipeline(Thread):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.params_general = params['general']
        params_logging = self.params_general.get('logging', {})
        log_filename = params_logging.get('filename', 'log/muvis-align.log')
        log_format = params_logging.get('format', '%(asctime)s %(levelname)s: %(message)s')
        self.verbose = params_logging.get('verbose', False)
        self.debug = params_logging.get('debug', False)
        init_logging(log_filename, log_format, verbose=self.verbose, debug=self.debug)

    def run(self):
        break_on_error = self.params_general.get('break_on_error', False)
        for operation_params in tqdm(self.params['operations']):
            error = False
            input_path = operation_params['input']
            if isinstance(input_path, dict):
                input_path = input_path.get('path')
            logging.info(f'Input: {input_path}')
            try:
                self.run_operation(operation_params)
            except Exception as e:
                logging.exception(f'Error processing: {input_path}')
                print(f'Error processing: {input_path}: {e}')
                error = True

            if error and break_on_error:
                break

        logging.info('Done!')

    def run_operation(self, params):
        operation = params['operation']
        use_global_metadata = 'global' in params.get('source_metadata', '')
        metadata_summary = self.params_general.get('metadata_summary', False)

        if isinstance(params['input'], dict):
            path = params['input'].get('path')
        else:
            path = params['input']
        filenames = sorted(dir_regex(path), key=lambda file: list(find_all_numbers(file)))    # sort first key first
        if len(filenames) == 0:
            logging.warning(f'Skipping operation {operation} (no files)')
            return False
        elif self.verbose:
            logging.info(f'# total files: {len(filenames)}')

        operation_parts = operation.split()
        if 'match' in operation_parts:
            # check if match label provided
            index = operation_parts.index('match') + 1
            if index < len(operation_parts):
                match_label = operation_parts[index]
            else:
                match_label = None
            matches = {}
            for filename in filenames:
                match_value = find_target_numeric(filename, match_label)
                if match_value is not None:
                    if match_value not in matches:
                        matches[match_value] = []
                    matches[match_value].append(filename)
            if len(matches) == 0:
                matches[0] = filenames
            filesets = []
            fileset_labels = []
            for label in sorted(matches):
                filesets.append(matches[label])
                fileset_labels.append(f'{match_label}:{label}')
            logging.info(f'# matched file sets: {len(filesets)}')
        else:
            filesets = [filenames]
            fileset_labels = [get_filetitle(filename) for filename in filenames]

        metadatas = []
        rotations = []
        global_center = None
        if metadata_summary or use_global_metadata:
            for fileset, fileset_label in zip(filesets, fileset_labels):
                metadata = get_images_metadata(fileset, params.get('source_metadata'))
                if metadata_summary:
                    logging.info(f'File set: {fileset_label} metadata:\n' + metadata['summary'])
                metadatas.append(metadata)
            if use_global_metadata:
                global_center = {dim: np.mean([metadata['center'][dim] for metadata in metadatas]) for dim in metadatas[0]['center']}
                rotations = [metadata['rotation'] for metadata in metadatas]
                # fix missing rotation values
                rotations = pd.Series(rotations).interpolate(limit_direction='both').to_numpy()

        n_set_workers = params.get('n_set_workers', 1)
        if n_set_workers > 1:
            value_sets = []
            for index, (fileset, fileset_label) in enumerate(zip(filesets, fileset_labels)):
                center = global_center if use_global_metadata else None
                rotation = rotations[index] if use_global_metadata else None
                value_sets.append({'fileset_label': fileset_label,
                                   'fileset': fileset,
                                   'params': params,
                                   'center': center,
                                   'rotation': rotation})

            with ThreadPoolExecutor(max_workers=n_set_workers) as executor:
                oks = executor.map(lambda kwargs: self.run_operation_thread(**kwargs), value_sets)
            return np.all([ok for ok in oks])
        else:
            ok = False
            for index, (fileset, fileset_label) in enumerate(zip(filesets, fileset_labels)):
                center = global_center if use_global_metadata else None
                rotation = rotations[index] if use_global_metadata else None
                ok |= self.run_operation_thread(fileset_label=fileset_label, fileset=fileset, params=params,
                                                 center=center, rotation=rotation)
            return ok

    def run_operation_thread(self, fileset_label, fileset, params, center, rotation):
        if fileset_label:
            logging.info(f'File set: {fileset_label}')

        mvs_registration = MVSRegistration()
        mvs_registration.init_params(params_general=self.params_general, params=params,
                                     label=fileset_label, input_path=fileset,
                                     global_center=center, global_rotation=rotation)
        return mvs_registration.run()
