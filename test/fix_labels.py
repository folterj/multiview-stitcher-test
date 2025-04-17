from src.util import import_json, export_json


if __name__ == '__main__':
    folder = 'D:/slides/EM04654_slice011/aligned_hpc/'
    filename = folder + 'mappings.json'
    out_filename = folder + 'mappings2.json'
    data = import_json(filename)
    data2 = {}
    for key, value in data.items():
        keys = key.split('-')
        key2 = f'slice{int(keys[1]):05}'
        data2[key2] = value
    export_json(out_filename, data2)
