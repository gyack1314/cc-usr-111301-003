"""按tile的GeoTIFF窗口采集测试。

小图全部tile逐像素与完整解码对照；跨tile、边缘padding、大端16位、Deflate
与水平预测器均覆盖。另检查目录拒绝、读取范围与缺块行为。
"""
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from capture_data.io import Source, load
from capture_data.tiff import (ImageSource, MissingBlock, TiffError,
                               Unsupported, open_tiff)
from tests.tiff_builder import (assemble, build_tiff, multi_image,
                                one_tile_blob, single_image, tile_bytes)

DATA = Path('data')


def source_of(blob, expected=None):
    return ImageSource.from_bytes(blob, expected)


def full_pixels(image):
    return image.capture((0, image.height, 0, image.width))


def _unsupported_factory(tags):
    """把标记为None的偏移/计数自动绑定到payload。"""

    def factory(offsets, counts):
        out = []
        for tag, typ, cnt, values in tags:
            if values is None:
                values = offsets if tag == 324 else counts
            out.append((tag, typ, cnt, list(values)))
        return out
    return factory


class SmallImageEquivalence(unittest.TestCase):
    def _check(self, blob, pixels, width, height, spp, depth, order):
        doc = open_tiff(source_of(blob))
        image = doc.select(0)
        got = full_pixels(image)
        self.assertEqual(got.rows, height, '窗口行数应等于图像行数')
        self.assertEqual(got.cols, width, '窗口列数应等于图像列数')
        self.assertEqual(got.samples_per_pixel, spp)
        self.assertEqual(got.depth, depth)
        self.assertEqual(got.byte_order, 'big' if order == '>' else 'little')
        expected = [v for px in pixels for v in px]
        self.assertEqual(list(got.values), expected,
                         '全窗口解码结果应与逐像素参考完全一致')
        # 每个像素也可按原图坐标单独取回
        for r in range(height):
            for c in range(width):
                self.assertEqual(got.pixel(r, c), pixels[r * width + c])

    def test_big_endian_16bit_predictor(self):
        blob, pixels, _, _ = single_image(3, 2, 2, 2, '>', spp=2, depth=16,
                                          compression=8, predictor=2)
        self._check(blob, pixels, 3, 2, 2, 16, '>')

    def test_little_endian_16bit_predictor(self):
        blob, pixels, _, _ = single_image(5, 3, 2, 2, '<', spp=1, depth=16,
                                          compression=8, predictor=2)
        self._check(blob, pixels, 5, 3, 1, 16, '<')

    def test_8bit_multi_band(self):
        blob, pixels, _, _ = single_image(4, 3, 2, 1, '>', spp=3, depth=8,
                                          compression=8, predictor=1)
        self._check(blob, pixels, 4, 3, 3, 8, '>')

    def test_8bit_predictor(self):
        blob, pixels, _, _ = single_image(4, 3, 2, 1, '>', spp=3, depth=8,
                                          compression=8, predictor=2)
        self._check(blob, pixels, 4, 3, 3, 8, '>')

    def test_uncompressed_16bit(self):
        blob, pixels, _, _ = single_image(3, 3, 2, 2, '<', spp=1, depth=16,
                                          compression=1, extra_tags=())
        self._check(blob, pixels, 3, 3, 1, 16, '<')

    def test_predictor_without_compression_is_rejected(self):
        # Predictor=2只与Compression=8一起接受；此处应为Unsupported
        raw_tile = struct.pack('>' + 'H' * 4, 0, 0, 0, 0)
        tags = [(256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
                (259, 3, 1, [1]), (262, 3, 1, [1]), (317, 3, 1, [2]),
                (322, 4, 1, [2]), (323, 4, 1, [2]), (324, 4, 1, [16]),
                (325, 4, 1, [len(raw_tile)])]
        blob, _ = build_tiff('>', [{'tags': tags}], raw_tile)
        with self.assertRaises(Unsupported):
            open_tiff(source_of(blob)).select(0)


class WindowSelection(unittest.TestCase):
    def test_small_window_reads_only_needed_tiles(self):
        blob, pixels, offsets, counts = single_image(5, 5, 2, 2, '>', 1, 16,
                                                     compression=8,
                                                     predictor=2)
        doc = open_tiff(source_of(blob))
        image = doc.select(0)
        capture = image.capture((1, 2, 3, 4))
        self.assertEqual([t['tile_index'] for t in capture.tiles], [1])
        self.assertEqual(capture.tile_reads,
                         [{'start': offsets[1], 'length': counts[1],
                           'sha256': capture.directory['source']['sha256'],
                           'bytes_sha256': hashlib.sha256(blob[offsets[1]:offsets[1] + counts[1]]).hexdigest()}])
        self.assertEqual(capture.tile_bytes_read, counts[1])
        self.assertEqual(capture.pixel(1, 3), pixels[1 * 5 + 3])
        for row in range(5):
            for col in range(5):
                if row == 1 and col == 3:
                    continue
                with self.assertRaises(ValueError):
                    capture.pixel(row, col)

    def test_cross_tile_window(self):
        blob, pixels, _, _ = single_image(5, 4, 2, 2, '<', 1, 16, 8, 1)
        image = open_tiff(source_of(blob)).select(0)
        window = (1, 4, 1, 4)
        capture = image.capture(window)
        expected = [v for r in range(1, 4) for c in range(1, 4)
                    for v in pixels[r * 5 + c]]
        self.assertEqual(list(capture.values), expected)
        # 3x2 tile网格中，行1..3与列1..3覆盖第0、1、2、3号tile
        # 5x4图、2x2 tile：窗口列1..4覆盖tile列0和1，行1..4覆盖tile行0和1
        self.assertEqual([t['tile_index'] for t in capture.tiles], [0, 1, 3, 4])

    def test_edge_padding_not_in_window(self):
        blob, pixels, _, _ = single_image(3, 2, 2, 2, '>', 2, 16, 8, 2,
                                          seed=4, edge_pad=99)
        image = open_tiff(source_of(blob)).select(0)
        capture = image.capture((1, 2, 2, 3))
        self.assertEqual(capture.pixel(1, 2), pixels[1 * 3 + 2],
                         '边缘padding不得进入窗口')

    def test_window_validation(self):
        blob, _, _, _ = single_image(4, 4, 2, 2)
        image = open_tiff(source_of(blob)).select(0)
        for bad in ((0, 0, 0, 1), (0, 5, 0, 1), (0, 2, 0, 5), (0, 2, 2, 1),
                    (True, 2, 0, 1), (0.0, 2, 0, 1), (0, 2, 0)):
            with self.subTest(window=bad):
                with self.assertRaises(ValueError):
                    image.capture(bad)

class DirectoryRules(unittest.TestCase):
    def test_byte_order_and_magic(self):
        for order in ('<', '>'):
            blob, _, _, _ = single_image(2, 2, 2, 2, order)
            doc = open_tiff(source_of(blob))
            self.assertEqual(doc.byte_order, 'little' if order == '<' else 'big')
        bad = bytearray((DATA / 'big-endian.tif').read_bytes())
        bad[:2] = b'XX'
        with self.assertRaises(Unsupported):
            open_tiff(source_of(bytes(bad)))
        bigtiff = struct.pack('>2sHHI', b'MM', 43, 8, 0)
        with self.assertRaises(Unsupported):
            open_tiff(source_of(bigtiff))
        wrong_magic = struct.pack('>2sHII', b'MM', 41, 8, 0)
        with self.assertRaises(TiffError):
            open_tiff(source_of(wrong_magic))

    def test_ifd_chain_multiple_images_and_explicit_selection(self):
        # 第0个IFD是2x2缩略图，第1个是6x4主图；必须显式选第1个
        thumb_payloads, _ = tile_bytes(2, 2, 2, 2, 1, 16, 8, 1, '>', 0, 7)
        main_payloads, main_pixels = tile_bytes(6, 4, 2, 2, 1, 16, 8, 2, '>',
                                                0, 7)
        blob, offsets = multi_image('>', [
            {'payloads': thumb_payloads, 'width': 2, 'height': 2,
             'tile_w': 2, 'tile_h': 2},
            {'payloads': main_payloads, 'width': 6, 'height': 4,
             'tile_w': 2, 'tile_h': 2, 'predictor': 2},
        ])
        doc = open_tiff(source_of(blob))
        self.assertEqual([s['summary']['width'] for s in doc.images()], [2, 6])
        self.assertEqual(len(doc.images()), 2)
        thumb = doc.select(0)
        main = doc.select(1)
        self.assertEqual((main.width, main.height), (6, 4))
        main_capture = full_pixels(main)
        expected = [v for px in main_pixels for v in px]
        self.assertEqual(list(main_capture.values), expected)
        self.assertEqual([t['tile_index'] for t in main_capture.tiles],
                         list(range(len(main_payloads))))
        # 缩略图与主图各自只绑定自己的tile表
        self.assertNotEqual(thumb.plan()['tile_offsets'],
                            main.plan()['tile_offsets'])
        with self.assertRaises(ValueError):
            doc.select(2)
        with self.assertRaises(ValueError):
            doc.select(True)

    def test_ifd_cycle_and_out_of_range(self):
        tags = [(256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
                (259, 3, 1, [8]), (262, 3, 1, [1]), (322, 4, 1, [2]),
                (323, 4, 1, [2]), (324, 4, 1, [0]), (325, 4, 1, [0])]
        payloads, _ = tile_bytes(2, 2, 2, 2, 1, 16, 8, 1, '>')
        _, tail = build_tiff('>', [{'tags': tags}], b''.join(payloads))
        offsets = []
        cursor = tail
        for p in payloads:
            offsets.append(cursor)
            cursor += len(p)
        tags = [(256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
                (259, 3, 1, [8]), (262, 3, 1, [1]), (322, 4, 1, [2]),
                (323, 4, 1, [2]), (324, 4, 1, [offsets[0]]),
                (325, 4, 1, [len(payloads[0])])]
        blob, _ = build_tiff('>', [{'tags': tags, 'next': 8}],
                             b''.join(payloads))
        with self.assertRaises(TiffError) as ctx:
            open_tiff(source_of(blob))
        self.assertIn('环', str(ctx.exception))
        blob2, _ = build_tiff('>', [{'tags': tags, 'next': 4000}],
                              b''.join(payloads))
        with self.assertRaises(TiffError):
            open_tiff(source_of(blob2))

    def test_value_offsets_not_guessed(self):
        # TileOffsets值指向越界位置时拒绝，不把偏移当成像素位置
        payloads, _ = tile_bytes(2, 2, 2, 2, 1, 16, 8, 1, '>')
        base = [(256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
                (259, 3, 1, [8]), (262, 3, 1, [1]), (322, 4, 1, [2]),
                (323, 4, 1, [2]), (324, 4, 2, [999999, 16]),
                (325, 4, 2, [len(payloads[0]), len(payloads[0])])]
        blob, _ = build_tiff('>', [{'tags': base}], b''.join(payloads))
        with self.assertRaises(TiffError):
            open_tiff(source_of(blob)).select(0)
        # tag 324的值偏移本身越界（外部块偏移指向文件外）
        bad = bytearray(blob)
        # 找到外部TileOffsets块的条目位置并改写其偏移
        n = struct.unpack('>H', bad[8:10])[0]
        for i in range(n):
            pos = 10 + 12 * i
            if struct.unpack('>H', bad[pos:pos + 2])[0] == 324:
                bad[pos + 8:pos + 12] = struct.pack('>I', len(bad) + 100)
                break
        with self.assertRaises(TiffError):
            open_tiff(source_of(bytes(bad)))

    def test_duplicate_key_tag_rejected(self):
        payloads, _ = tile_bytes(2, 2, 2, 2, 1, 16, 8, 1, '>')
        blob, tail = build_tiff('>', [{'tags': [
            (256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
            (259, 3, 1, [8]), (259, 3, 1, [8]), (262, 3, 1, [1]),
            (322, 4, 1, [2]), (323, 4, 1, [2]), (324, 4, 1, [0]),
            (325, 4, 1, [0])]}], b''.join(payloads))
        with self.assertRaises(TiffError) as ctx:
            open_tiff(source_of(blob))
        self.assertIn('重复', str(ctx.exception))

    def test_inline_and_offset_field_values(self):
        blob, _, _, _ = single_image(4, 4, 2, 2, '<', 2, 16, 8, 2,
                                     nodata='65535', geo=True)
        doc = open_tiff(source_of(blob))
        entries = doc.select(0).ifd['entries']
        self.assertEqual(entries[256]['location'], 'inline')
        self.assertTrue(entries[324]['location'].startswith('offset:'))
        self.assertEqual(entries[324]['count'], 4)
        self.assertEqual(entries[324]['bytes'], 16)
        self.assertEqual(entries[325]['count'], 4)
        geo = doc.select(0).geo_tags()
        self.assertEqual(geo['ModelPixelScale']['value'], [1.0, 1.0, 0.0])
        self.assertEqual(geo['ModelTiepoint']['type'], 12)
        self.assertEqual(geo['GeoKeyDirectory']['value'],
                         [1, 1, 0, 1, 1024, 0, 1, 1])
        self.assertEqual(geo['GeoAsciiParams']['type'], 2)

    def test_nodata_is_metadata_not_fill(self):
        blob, pixels, _, _ = single_image(2, 2, 2, 2, '>', 1, 16, 8, 1,
                                          nodata='65535')
        image = open_tiff(source_of(blob)).select(0)
        meta = image.nodata_metadata()
        self.assertEqual(meta['value'], '65535\x00')
        self.assertEqual(meta['source'], 'tag 42113 GDAL_NODATA')
        self.assertFalse(meta['is_fill_value'])
        self.assertEqual(meta['value'], '65535\x00')
        capture = image.capture((0, 1, 0, 1))
        self.assertEqual(capture.pixel(0, 0), pixels[0],
                         'NoData不得覆盖真实像素')
        self.assertEqual(capture.nodata['source'], 'tag 42113 GDAL_NODATA')

    def test_unsupported_codings(self):
        # PlanarConfiguration=2明确不支持
        payloads, _ = tile_bytes(4, 4, 2, 2, 2, 16, 8, 1, '>')
        factory = _unsupported_factory(
            [(256, 4, 1, [4]), (257, 4, 1, [4]), (258, 3, 2, [16, 16]),
             (259, 3, 1, [8]), (262, 3, 1, [1]), (277, 3, 1, [2]),
             (284, 3, 1, [2]), (322, 4, 1, [2]), (323, 4, 1, [2]),
             (324, 4, 4, None), (325, 4, 4, None)])
        blob, _ = assemble('>', [(factory, payloads)])
        with self.assertRaises(Unsupported) as ctx:
            open_tiff(source_of(blob)).select(0)
        self.assertIn('PlanarConfiguration', str(ctx.exception))
        # Compression=5（LZW）明确不支持
        factory = _unsupported_factory(
            [(256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
             (259, 3, 1, [5]), (262, 3, 1, [1]), (322, 4, 1, [2]),
             (323, 4, 1, [2]), (324, 4, 1, None), (325, 4, 1, None)])
        payloads, _ = tile_bytes(2, 2, 2, 2, 1, 16, 1, 1, '>')
        blob, _ = assemble('>', [(factory, payloads)])
        with self.assertRaises(Unsupported) as ctx:
            open_tiff(source_of(blob)).select(0)
        self.assertIn('Compression', str(ctx.exception))
        # 有符号16位明确不支持
        factory = _unsupported_factory(
            [(256, 4, 1, [2]), (257, 4, 1, [2]), (258, 3, 1, [16]),
             (259, 3, 1, [8]), (262, 3, 1, [1]), (339, 3, 1, [2]),
             (322, 4, 1, [2]), (323, 4, 1, [2]), (324, 4, 1, None),
             (325, 4, 1, None)])
        blob, _ = assemble('>', [(factory, payloads)])
        with self.assertRaises(Unsupported):
            open_tiff(source_of(blob)).select(0)

    def test_strip_image_rejected(self):
        tags = [(256, 4, 1, [4]), (257, 4, 1, [4]), (258, 3, 1, [16]),
                (259, 3, 1, [1]), (262, 3, 1, [1]), (273, 4, 1, [8]),
                (277, 3, 1, [1]), (278, 4, 1, [4]), (279, 4, 1, [32])]
        blob, _ = build_tiff('>', [{'tags': tags}], b'\0' * 32)
        with self.assertRaises(Unsupported):
            open_tiff(source_of(blob)).select(0)


class DecodeRules(unittest.TestCase):
    def test_deflate_exact_length(self):
        # 单tile 16位，预期解压16字节
        over = zlib.compress(b'\0' * 9)
        short = zlib.compress(b'\0' * 7)
        with self.assertRaises(TiffError) as ctx:
            open_tiff(source_of(one_tile_blob('>', over))).select(0).capture(
                (0, 2, 0, 2))
        self.assertIn('超过预期', str(ctx.exception))
        with self.assertRaises(TiffError) as ctx:
            open_tiff(source_of(one_tile_blob('>', short))).select(0).capture(
                (0, 2, 0, 2))
        self.assertIn('预期', str(ctx.exception))
        # 解压出正好16字节则成功
        exact = zlib.compress(bytes(range(8)))
        capture = open_tiff(source_of(one_tile_blob('>', exact))).select(0).capture(
            (0, 2, 0, 2))
        self.assertEqual(len(capture.values), 4)

    def test_corrupt_deflate_reports_missing_tile(self):
        ref = json.loads((DATA / 'corrupt-compressed-tile.json').read_text())
        source = Source(DATA, ref['source'])
        doc = open_tiff(ImageSource.from_bytes(source.raw, ref['source']['sha256']))
        image = doc.select(ref['ifd_index'])
        w = ref['window']
        with self.assertRaises(MissingBlock) as ctx:
            image.capture((w['row_start'], w['row_stop'],
                           w['col_start'], w['col_stop']),
                          ref['max_decoded_tile_bytes'])
        self.assertEqual([t['tile_index'] for t in ctx.exception.tiles], [1])
        self.assertIn('Deflate', str(ctx.exception))
        self.assertEqual(source.raw, bytes(source.raw),
                         '源对象不得被修改')
        # 不含任何填零像素：失败时没有结果对象
        self.assertNotIsInstance(ctx.exception, type(None))

    def test_predictor_row_reset(self):
        # 构造仅含Predictor=1与=2的两种编码，全窗口结果应相同
        a, pixels, _, _ = single_image(4, 4, 4, 4, '>', 1, 16, 8, 1)
        b, pixels_b, _, _ = single_image(4, 4, 4, 4, '>', 1, 16, 8, 2)
        self.assertEqual(pixels, pixels_b)
        ca = full_pixels(open_tiff(source_of(a)).select(0))
        cb = full_pixels(open_tiff(source_of(b)).select(0))
        self.assertEqual(list(ca.values), list(cb.values))

    def test_16bit_addition_respects_file_order(self):
        # 同一数值用II/MM编码，解码结果必须一致且加法不发生字节序错位
        le, pixels, _, _ = single_image(3, 3, 2, 2, '<', 1, 16, 8, 2)
        be, _, _, _ = single_image(3, 3, 2, 2, '>', 1, 16, 8, 2)
        self.assertEqual(list(full_pixels(open_tiff(source_of(le)).select(0)).values),
                         list(full_pixels(open_tiff(source_of(be)).select(0)).values))
        self.assertEqual(list(full_pixels(open_tiff(source_of(le)).select(0)).values),
                         [v for px in pixels for v in px])


class SourceRules(unittest.TestCase):
    def test_read_at_checks_returned_range(self):
        blob = (DATA / 'big-endian.tif').read_bytes()
        def short(start, length):
            return blob[start:start + max(0, length - 1)]
        source = ImageSource(hashlib.sha256(blob).hexdigest(), len(blob), short)
        with self.assertRaises(ValueError):
            source.read_at(0, 8)
        for start, length in ((-1, 1), (0, len(blob) + 1), (True, 1), (0, -1)):
            with self.subTest(start=start, length=length):
                with self.assertRaises(ValueError):
                    source.read_at(start, length)

    def test_digest_binding(self):
        blob = (DATA / 'big-endian.tif').read_bytes()
        good = hashlib.sha256(blob).hexdigest()
        with self.assertRaises(ValueError):
            ImageSource.from_bytes(blob, '0' * 64)
        source = ImageSource.from_bytes(blob, good)
        self.assertEqual(source.identify(), {'sha256': good, 'length': len(blob)})
        doc = open_tiff(source)
        plan = doc.directory_plan()
        self.assertEqual(plan['source']['sha256'], good)
        self.assertEqual(len(plan['directory_ranges']), len(source.calls))
        for call in plan['directory_ranges']:
            self.assertEqual(call['sha256'], good)

    def test_cache_is_bound_to_identity(self):
        blob = (DATA / 'big-endian.tif').read_bytes()
        doc_a = open_tiff(source_of(blob))
        cap_a = doc_a.select(0).capture((0, 1, 0, 1))
        blob_b = (DATA / 'bad-deflate.tif').read_bytes()
        doc_b = open_tiff(source_of(blob_b))
        with self.assertRaises(MissingBlock):
            doc_b.select(0).capture((0, 2, 1, 3))
        self.assertNotEqual(cap_a.plan['source']['sha256'],
                            doc_b.directory_plan()['source']['sha256'])
        # 同一Image内重复窗口可复用缓存，但换对象仍重新读取
        image = doc_a.select(0)
        image.capture((0, 2, 1, 3))
        before = len(doc_a.source.calls)
        image.capture((0, 2, 1, 3))
        self.assertEqual(len(doc_a.source.calls), before)
        other = open_tiff(source_of(blob))
        other.select(0).capture((0, 2, 1, 3))
        self.assertGreater(len(other.source.calls), 0)
        self.assertGreater(len(doc_a.source.calls), 0)
        self.assertNotEqual(id(doc_a.source), id(other.source))

    def test_manifest_fixtures_full_decode(self):
        normal = json.loads((DATA / 'normal.json').read_text())
        source = Source(DATA, normal['source'])
        doc = open_tiff(ImageSource.from_bytes(source.raw, normal['source']['sha256']))
        image = doc.select(normal['ifd_index'])
        w = normal['window']
        capture = image.capture((w['row_start'], w['row_stop'],
                                 w['col_start'], w['col_stop']),
                                normal['max_decoded_tile_bytes'])
        self.assertEqual(list(capture.values), [2, 20, 3, 30, 5, 50, 6, 60])
        self.assertEqual(capture.geo_tags['GeoKeyDirectory']['value'],
                         [1, 1, 0, 1, 1024, 0, 1, 1])
        self.assertEqual(capture.nodata['value'], '65535\x00')
        self.assertEqual(capture.plan['window_to_tiles'][0]['source_range'],
                         [268, 287])
        edge = json.loads((DATA / 'edge-padding-window.json').read_text())
        source = Source(DATA, edge['source'])
        doc = open_tiff(ImageSource.from_bytes(source.raw, edge['source']['sha256']))
        image = doc.select(edge['ifd_index'])
        w = edge['window']
        capture = image.capture((w['row_start'], w['row_stop'],
                                 w['col_start'], w['col_stop']),
                                edge['max_decoded_tile_bytes'])
        self.assertEqual(list(capture.values), [6, 60])
        self.assertEqual([t['tile_index'] for t in capture.tiles], [1])


if __name__ == '__main__':
    unittest.main()
