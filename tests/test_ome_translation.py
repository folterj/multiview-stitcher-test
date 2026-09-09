"""extract_ome_translation must not scale with the size of the whole dataset.

A multi-file OME-TIFF repeats the entire dataset's OME XML in every file's header, so parsing
all of it per file to read one plane's position is O(files^2) overall. The streaming parse must
give byte-for-byte the same answers as the xml2dict version it replaces - including the {} it
returns for a multi-Image XML - while stopping early.
"""
import pytest
from tifffile import tifffile

from muvis_align.image.ome_tiff_helper import extract_ome_translation_from_xml

NS = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'


def ome_xml(images):
    """images: list of dicts of Plane attributes (or None for an Image with no Plane)."""
    body = []
    for index, plane in enumerate(images):
        plane_xml = ''
        if plane is not None:
            attrs = ' '.join(f'{key}="{value}"' for key, value in plane.items())
            plane_xml = f'<Plane TheC="0" TheZ="0" TheT="0" {attrs}/>'
        body.append(f'<Image ID="Image:{index}"><Pixels ID="Pixels:{index}" Type="uint16"'
                    f' SizeX="64" SizeY="64" SizeC="1" SizeZ="1" SizeT="1">'
                    f'{plane_xml}</Pixels></Image>')
    return (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}">'
            + ''.join(body) + '</OME>')


def reference_implementation(ome_metadata):
    """The xml2dict version this replaces, kept here as the oracle."""
    metadata = tifffile.xml2dict(ome_metadata)
    if 'OME' in metadata:
        metadata = metadata['OME']
    if 'Image' in metadata and 'Pixels' in metadata['Image'] and 'Plane' in metadata['Image']['Pixels']:
        plane_metadata = metadata['Image']['Pixels']['Plane']
        if isinstance(plane_metadata, list):
            plane_metadata = plane_metadata[0]
        position = {}
        for dim in ['X', 'Y', 'Z']:
            key = f'Position{dim}'
            if key in plane_metadata:
                from muvis_align.util import convert_to_um
                position[dim.lower()] = convert_to_um(float(plane_metadata[key]),
                                                      plane_metadata.get(f'{key}Unit', 'um'))
        return position
    return {}


CASES = [
    ('xy um', [{'PositionX': 12.5, 'PositionY': -3.25,
                'PositionXUnit': 'um', 'PositionYUnit': 'um'}]),
    ('xyz', [{'PositionX': 1.0, 'PositionY': 2.0, 'PositionZ': 3.0,
              'PositionXUnit': 'um', 'PositionYUnit': 'um', 'PositionZUnit': 'um'}]),
    ('no unit attributes', [{'PositionX': 7.0, 'PositionY': 8.0}]),
    ('millimetre units', [{'PositionX': 1.5, 'PositionY': 2.5,
                           'PositionXUnit': 'mm', 'PositionYUnit': 'mm'}]),
    ('x only', [{'PositionX': 4.0}]),
    ('no positions', [{}]),
    ('no plane', [None]),
    # multi-image: the historical behaviour is no position at all
    ('two images', [{'PositionX': 1.0, 'PositionY': 2.0}, {'PositionX': 3.0, 'PositionY': 4.0}]),
    ('many images', [{'PositionX': float(i), 'PositionY': 0.0} for i in range(20)]),
]


@pytest.mark.parametrize('label, images', CASES)
def test_matches_the_xml2dict_implementation(label, images):
    xml = ome_xml(images)
    assert extract_ome_translation_from_xml(xml) == reference_implementation(xml)


def test_multi_image_xml_yields_no_position():
    xml = ome_xml([{'PositionX': 1.0, 'PositionY': 2.0}] * 3)
    assert extract_ome_translation_from_xml(xml) == {}


def test_units_are_converted_to_um():
    xml = ome_xml([{'PositionX': 1.5, 'PositionY': 2.5,
                    'PositionXUnit': 'mm', 'PositionYUnit': 'mm'}])
    assert extract_ome_translation_from_xml(xml) == pytest.approx({'x': 1500.0, 'y': 2500.0})


def test_parsing_stops_once_the_answer_is_settled():
    """The whole point: cost must not grow with the dataset, since a multi-file OME-TIFF carries
    the whole dataset's XML in every file's header.

    Asserted structurally rather than by the clock - a wall-time bound is flaky on a loaded
    machine, and it was. The document below is malformed well past the first fed chunk, so
    parsing it to the end would raise; returning normally proves the parse stopped early.
    """
    images = [{'PositionX': float(i), 'PositionY': 0.0} for i in range(4000)]
    document = ome_xml(images).replace('</OME>', '<Unclosed>' * 5 + '<<<not xml&&&</OME>')
    assert len(document) > 512 * 1024, 'the malformed tail must sit beyond the first chunk'

    # multi-Image, so no position - but crucially, no parse error either
    assert extract_ome_translation_from_xml(document) == {}


def test_a_single_image_document_is_still_read_to_the_end():
    # the early exit fires on a second <Image>; with only one there is nothing to stop at, and
    # such a document is small (a single tile's own metadata)
    xml = ome_xml([{'PositionX': 9.0, 'PositionY': 10.0}])
    assert extract_ome_translation_from_xml(xml) == pytest.approx({'x': 9.0, 'y': 10.0})


def test_single_image_xml_still_reads_its_plane():
    # the early exit must not fire for a genuinely single-image file
    xml = ome_xml([{'PositionX': 9.0, 'PositionY': 10.0}])
    assert extract_ome_translation_from_xml(xml) == pytest.approx({'x': 9.0, 'y': 10.0})
