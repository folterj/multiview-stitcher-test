import argparse
import yaml

from src.Pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'multiview-stitcher')
    parser.add_argument('--params',
                        help='The parameters file',
                        default='resources/params.yml')

    args = parser.parse_args()
    with open(args.params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    pipeline = Pipeline(params)
    pipeline.run()
