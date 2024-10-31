import yaml

from src.registration import init_logger, run_stitch_overlay


if __name__ == '__main__':
    with open('resources/params_test.yml', 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    params_general = params['general']
    init_logger(params_general)
    verbose = params_general.get('verbose', False)

    run_stitch_overlay('output', verbose=verbose)
