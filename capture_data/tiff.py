"""经典GeoTIFF的目录解析与按tile的窗口采集。

只接受经典TIFF（magic=42，II或MM字节序），明确拒绝BigTIFF。IFD链中的每个
图像彼此独立；调用者必须用明确的IFD序号选图，本模块不会把首个IFD或缩略图
默认为主图。

窗口由原图像素行列的半开矩形定义，不做地图投影重采样。采集只读取窗口实际
覆盖的tile字节，每个tile仍按完整tile尺寸解码，再按图像边界和窗口裁切。
GeoKey与变换tag按原值返回。NoData只是带来源的元数据，绝不用它填充未采集
区域；缺少所需tile时抛出 MissingBlock，不产生填零后的完整图像。
"""
import hashlib
import struct
import zlib
from array import array
from dataclasses import dataclass, field

# 经典TIFF字段类型及其字节长度
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1,
              7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_INT_FORMAT = {1: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i'}
_TAG_NAMES = {
    254: 'NewSubfileType', 256: 'ImageWidth', 257: 'ImageLength',
    258: 'BitsPerSample', 259: 'Compression', 262: 'PhotometricInterpretation',
    270: 'ImageDescription', 273: 'StripOffsets', 274: 'Orientation',
    277: 'SamplesPerPixel', 278: 'RowsPerStrip', 279: 'StripByteCounts',
    282: 'XResolution', 283: 'YResolution', 284: 'PlanarConfiguration',
    296: 'ResolutionUnit', 305: 'Software', 317: 'Predictor',
    320: 'ColorMap', 322: 'TileWidth', 323: 'TileLength',
    324: 'TileOffsets', 325: 'TileByteCounts', 339: 'SampleFormat',
    33550: 'ModelPixelScale', 33922: 'ModelTiepoint',
    34264: 'ModelTransformation', 34735: 'GeoKeyDirectory',
    34736: 'GeoDoubleParams', 34737: 'GeoAsciiParams', 42113: 'GDAL_NODATA',
}
# 关键像素布局tag；重复出现即视为目录损坏
_KEY_TAGS = {256, 257, 258, 259, 262, 277, 284, 317, 322, 323, 324, 325, 339}
_GEO_TAGS = (33550, 33922, 34264, 34735, 34736, 34737)
_NODATA_TAG = 42113

_MAX_COUNT = 1 << 28
_MAX_IFDS = 4096
_MAX_TILES = 1 << 22
_MAX_DIMENSION = 1 << 31
_MAX_IMAGE_PIXELS = 1 << 40
_MAX_DECODED_TILE_DEFAULT = 1 << 24
_MAX_READ_BUDGET_DEFAULT = 1 << 28


class TiffError(ValueError):
    """目录或像素数据不合法。"""


class Unsupported(TiffError):
    """文件使用本工具明确不支持的编码。"""


class MissingBlock(TiffError):
    """所需tile块读取或解码失败；结果中不含任何填充像素。"""

    def __init__(self, message, tiles):
        super().__init__(message)
        self.tiles = [dict(t) for t in tiles]


def _int(value, what):
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f'{what}必须是整数')
    return value


@dataclass(frozen=True)
class _Read:
    start: int
    length: int
    sha256: str
    bytes_sha256: str

    def as_dict(self):
        return {'start': self.start, 'length': self.length,
                'sha256': self.sha256, 'bytes_sha256': self.bytes_sha256}


class ImageSource:
    """绑定长度与对象摘要的只读字节源。

    read_at 每次返回真实区间，核验返回类型与长度。适配器不缓存，缓存只能出现在
    绑定同一对象身份与摘要的调用方。
    """

    def __init__(self, sha256, length, read_at):
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError('对象摘要必须是64位十六进制')
        try:
            int(sha256, 16)
        except ValueError:
            raise ValueError('对象摘要必须是64位十六进制') from None
        self._sha256 = sha256.lower()
        self._length = _int(length, '对象长度')
        if self._length < 0:
            raise ValueError('对象长度不能为负')
        if not callable(read_at):
            raise TypeError('read_at必须可调用')
        self._read_at = read_at
        self._calls = []

    @classmethod
    def from_bytes(cls, raw, sha256=None):
        """由不可变字节构造源；给定期望摘要时必须完全一致。"""
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError('字节源必须是bytes')
        data = bytes(raw)
        actual = hashlib.sha256(data).hexdigest()
        if sha256 is not None and sha256.lower() != actual:
            raise ValueError('对象摘要不符')
        return cls(actual, len(data), lambda start, length: data[start:start + length])

    @classmethod
    def from_source(cls, source):
        """适配已有read_at对象（例如capture_data.io.Source）。"""
        sha = getattr(source, 'sha256', None)
        read = getattr(source, 'read_at', None)
        if not callable(read):
            raise TypeError('源缺少read_at')
        if hasattr(source, 'length'):
            length = source.length
        elif hasattr(source, 'raw'):
            length = len(source.raw)
        else:
            raise TypeError('无法取得源长度')
        return cls(sha, length, read)

    @property
    def sha256(self):
        return self._sha256

    @property
    def length(self):
        return self._length

    @property
    def calls(self):
        return [r.as_dict() for r in self._calls]

    def identify(self):
        return {'sha256': self._sha256, 'length': self._length}

    def read_at(self, start, length):
        try:
            _int(start, '范围起点')
            _int(length, '范围长度')
        except TypeError as exc:
            raise ValueError(str(exc)) from None
        if start < 0 or length < 0 or start > self._length or \
                start + length > self._length:
            raise ValueError('范围越界')
        raw = self._read_at(start, length)
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError('源返回类型非法')
        if len(raw) != length:
            raise ValueError('源短读：返回区间与请求区间不符')
        data = bytes(raw)
        self._calls.append(_Read(start, length, self._sha256,
                                 hashlib.sha256(data).hexdigest()))
        return data

    def __len__(self):
        return self._length


@dataclass
class Doc:
    """已解析的IFD链；调用者按明确IFD序号选图。"""
    source: ImageSource
    order: str
    ifds: list
    directory_ranges: list
    directory_digest: str = ''

    @property
    def byte_order(self):
        return 'little' if self.order == '<' else 'big'

    def images(self):
        """列出全部IFD摘要；缩略图也如实列出，不默认当主图。"""
        return [{'ifd_index': row['index'], 'offset': row['offset'],
                 'summary': dict(row['summary']),
                 'warnings': list(row['warnings'])} for row in self.ifds]

    def select(self, ifd_index):
        if type(ifd_index) is not int:
            raise ValueError('IFD序号必须是整数')
        if not 0 <= ifd_index < len(self.ifds):
            raise ValueError('IFD序号超出范围')
        return Image(self, self.ifds[ifd_index])

    def directory_plan(self):
        """绑定源摘要、长度与全部目录读取区间。"""
        return {'source': self.source.identify(),
                'byte_order': self.byte_order,
                'directory_digest': self.directory_digest,
                'directory_ranges': [dict(r) for r in self.directory_ranges],
                'ifds': [{'ifd_index': r['index'], 'offset': r['offset'],
                          'entry_count': r['count']} for r in self.ifds]}


def open_tiff(source):
    """解析IFD链和字段，不读取任何tile像素字节。"""
    if not isinstance(source, ImageSource):
        raise TypeError('source必须是ImageSource')
    if len(source) < 8:
        raise TiffError('文件短于TIFF头')
    head = source.read_at(0, 8)
    if head[:2] == b'II':
        order = '<'
    elif head[:2] == b'MM':
        order = '>'
    else:
        raise Unsupported('字节序标记必须是II或MM')
    magic = struct.unpack(order + 'H', head[2:4])[0]
    if magic == 43:
        raise Unsupported('BigTIFF不支持')
    if magic != 42:
        raise TiffError('magic必须为42')
    offset = struct.unpack(order + 'I', head[4:8])[0]
    if offset == 0:
        raise TiffError('没有IFD')
    if offset < 8 or offset % 2 or offset + 2 > len(source):
        raise TiffError('首个IFD偏移越界或未对齐')
    chain = []
    seen = set()
    while offset:
        if len(chain) >= _MAX_IFDS:
            raise TiffError('IFD数量超限')
        if offset in seen:
            raise TiffError('IFD链存在环')
        if offset % 2 or offset < 8 or offset + 2 > len(source):
            raise TiffError('IFD偏移越界或未对齐')
        seen.add(offset)
        count = struct.unpack(order + 'H', source.read_at(offset, 2))[0]
        entry_end = offset + 2 + 12 * count
        if entry_end + 4 > len(source):
            raise TiffError('IFD条目区或下一IFD指针越界')
        block = source.read_at(offset + 2, 12 * count + 4)
        next_offset = struct.unpack(order + 'I', block[12 * count:])[0]
        chain.append({'index': len(chain), 'offset': offset,
                      'count': count, 'block': block})
        if next_offset:
            if next_offset in seen:
                raise TiffError('IFD链存在环')
            if next_offset % 2 or next_offset < 8 or next_offset + 2 > len(source):
                raise TiffError('下一IFD偏移越界或未对齐')
        offset = next_offset
    ifds = []
    for row in chain:
        ifds.append(_parse_ifd(source, order, row, chain))
    directory_ranges = [dict(c) for c in source.calls]
    digest = hashlib.sha256()
    for call in source.calls:
        digest.update(f"{call['start']}:{call['length']}:{call['bytes_sha256']}\n".encode('ascii'))
    return Doc(source, order, ifds, directory_ranges, digest.hexdigest())


def _parse_ifd(source, order, row, chain):
    block = row['block']
    entries = {}
    warnings = []
    # 先把本IFD内所有越界值读出来，便于按段落判断重读
    for i in range(row['count']):
        tag, typ, cnt = struct.unpack(order + 'HHI', block[12 * i:12 * i + 8])
        if typ not in _TYPE_SIZE:
            raise TiffError(f'tag{tag}字段类型{typ}未知')
        if cnt > _MAX_COUNT:
            raise TiffError(f'tag{tag}计数超限')
        size = _TYPE_SIZE[typ] * cnt
        payload = block[12 * i + 8:12 * i + 12]
        if size <= 4:
            data = payload[:size]
            location = 'inline'
        else:
            value_offset = struct.unpack(order + 'I', payload)[0]
            if value_offset < 8 or value_offset > len(source) or \
                    size > len(source) or value_offset + size > len(source):
                raise TiffError(f'tag{tag}值偏移或长度越界')
            data = source.read_at(value_offset, size)
            location = f'offset:{value_offset}'
        entry = {'tag': tag, 'type': typ, 'count': cnt,
                 'type_size': _TYPE_SIZE[typ], 'bytes': size,
                 'location': location, 'data': data}
        if tag in entries:
            if tag in _KEY_TAGS:
                raise TiffError(f'关键tag{tag}重复')
            warnings.append(f'重复非关键tag{tag}，保留首次')
            continue
        entries[tag] = entry
    return {'index': row['index'], 'offset': row['offset'],
            'count': row['count'], 'entries': entries,
            'warnings': warnings,
            'summary': _summarize_ifd(order, entries)}


def _values(order, entry):
    typ, cnt = entry['type'], entry['count']
    if typ == 2:
        return entry['data'].decode('latin-1')
    if typ in _INT_FORMAT:
        code = _INT_FORMAT[typ]
        return list(struct.unpack(order + code * cnt, entry['data'])) if cnt else []
    if typ == 5:
        nums = struct.unpack(order + 'I' * (2 * cnt), entry['data']) if cnt else ()
        return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
    if typ == 10:
        nums = struct.unpack(order + 'i' * (2 * cnt), entry['data']) if cnt else ()
        return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
    if typ == 11:
        return list(struct.unpack(order + 'f' * cnt, entry['data'])) if cnt else []
    if typ == 12:
        return list(struct.unpack(order + 'd' * cnt, entry['data'])) if cnt else []
    return entry['data'].hex()


def _ints(order, entries, tag, expected_count=None, required=False):
    entry = entries.get(tag)
    if entry is None:
        if required:
            raise TiffError(f'缺少必需tag{tag}')
        return None
    if entry['type'] not in _INT_FORMAT:
        raise TiffError(f'tag{tag}字段类型{entry["type"]}不能作为整数')
    if expected_count is not None and entry['count'] != expected_count:
        raise TiffError(f'tag{tag}计数{entry["count"]}不等于{expected_count}')
    return _values(order, entry)


def _single(order, entries, tag, required=False):
    values = _ints(order, entries, tag, 1, required)
    return None if values is None else values[0]


def _ascii(entries, tag):
    entry = entries.get(tag)
    if entry is None or entry['type'] != 2:
        return None
    return entry['data'].decode('latin-1').rstrip('\x00')


def _raw_tag(order, entries, tag):
    entry = entries.get(tag)
    if entry is None:
        return None
    return {'tag': tag, 'name': _TAG_NAMES.get(tag, f'Tag{tag}'),
            'type': entry['type'], 'count': entry['count'],
            'bytes': entry['bytes'], 'location': entry['location'],
            'value': _values(order, entry)}


def _summarize_ifd(order, entries):
    def val(tag):
        try:
            return _single(order, entries, tag)
        except TiffError:
            return None

    bits = None
    entry = entries.get(258)
    if entry is not None and entry['type'] in _INT_FORMAT:
        bits = _values(order, entry)
    return {
        'width': val(256), 'height': val(257),
        'samples_per_pixel': val(277), 'bits_per_sample': bits,
        'compression': val(259), 'photometric': val(262),
        'planar_configuration': val(284), 'predictor': val(317),
        'tile_width': val(322), 'tile_height': val(323),
        'tiled': 324 in entries and 325 in entries,
        'subfile_type': val(254),
        'description': _ascii(entries, 270),
        'software': _ascii(entries, 305),
        'has_geokeys': 34735 in entries,
        'nodata_tag': _NODATA_TAG in entries,
    }


class Image:
    """选定且已校验编码的单个IFD；像素按需读取。"""

    def __init__(self, doc, ifd):
        self.doc = doc
        self.ifd = ifd
        self._plan = _make_plan(doc.order, ifd, len(doc.source))
        self._cache = {}

    @property
    def ifd_index(self):
        return self.ifd['index']

    @property
    def width(self):
        return self._plan['width']

    @property
    def height(self):
        return self._plan['length']

    def plan(self):
        p = self._plan
        return {'ifd_index': self.ifd['index'], 'ifd_offset': self.ifd['offset'],
                'width': p['width'], 'height': p['length'],
                'samples_per_pixel': p['spp'],
                'bits_per_sample': list(p['bps']),
                'byte_order': self.doc.byte_order,
                'compression': p['comp'], 'predictor': p['pred'],
                'planar_configuration': p['plan'],
                'photometric': p['photo'],
                'tile_width': p['tw'], 'tile_height': p['th'],
                'tile_columns': p['cols'], 'tile_rows': p['rows'],
                'expected_decoded_tile_bytes': p['expected'],
                'tile_offsets': list(p['offsets']),
                'tile_byte_counts': list(p['counts']),
                'directory_digest': self.doc.directory_digest,
                'source': self.doc.source.identify()}

    def geo_tags(self):
        """GeoKey与变换tag的原值，不做坐标换算。"""
        out = {}
        for tag in _GEO_TAGS:
            raw = _raw_tag(self.doc.order, self.ifd['entries'], tag)
            if raw is not None:
                out[raw['name']] = raw
        return out

    def nodata_metadata(self):
        """NoData仅作为带来源的元数据；绝不是可填充的像素值。"""
        raw = _raw_tag(self.doc.order, self.ifd['entries'], _NODATA_TAG)
        if raw is None:
            return None
        raw = dict(raw)
        raw['source'] = 'tag 42113 GDAL_NODATA'
        raw['is_fill_value'] = False
        return raw

    def capture(self, window, max_decoded_tile_bytes=_MAX_DECODED_TILE_DEFAULT,
                read_budget=_MAX_READ_BUDGET_DEFAULT):
        """按原图坐标采集窗口；只读取窗口覆盖的tile。"""
        row_start, row_stop, col_start, col_stop = _check_window(
            window, self.width, self.height)
        max_decoded_tile_bytes = _int(max_decoded_tile_bytes, '解码上限')
        read_budget = _int(read_budget, '读取预算')
        if max_decoded_tile_bytes < 0 or read_budget < 0:
            raise ValueError('上限不能为负')
        p = self._plan
        if p['expected'] > max_decoded_tile_bytes:
            raise Unsupported(
                f'完整tile需{p["expected"]}字节解码，超过上限{max_decoded_tile_bytes}')
        tr0, tr1 = row_start // p['th'], (row_stop - 1) // p['th']
        tc0, tc1 = col_start // p['tw'], (col_stop - 1) // p['tw']
        needed = []
        for tr in range(tr0, tr1 + 1):
            for tc in range(tc0, tc1 + 1):
                index = tr * p['cols'] + tc
                r0 = max(row_start, tr * p['th'])
                r1 = min(row_stop, (tr + 1) * p['th'], p['length'])
                c0 = max(col_start, tc * p['tw'])
                c1 = min(col_stop, (tc + 1) * p['tw'], p['width'])
                needed.append({
                    'tile_index': index, 'tile_row': tr, 'tile_col': tc,
                    'tile_image_row': tr * p['th'],
                    'tile_image_col': tc * p['tw'],
                    'source_start': p['offsets'][index],
                    'source_length': p['counts'][index],
                    'decoded_length': p['expected'],
                    'window_row_start': r0, 'window_row_stop': r1,
                    'window_col_start': c0, 'window_col_stop': c1,
                    'crop_row_offset': r0 - tr * p['th'],
                    'crop_col_offset': c0 - tc * p['tw'],
                    'crop_rows': r1 - r0, 'crop_cols': c1 - c0,
                })
        total_bytes = sum(t['source_length'] for t in needed)
        if total_bytes > read_budget:
            raise Unsupported(f'所需tile共{total_bytes}字节，超过读取预算{read_budget}')
        pixel_rows = row_stop - row_start
        pixel_cols = col_stop - col_start
        spp = p['spp']
        values = [None] * (pixel_rows * pixel_cols * spp)
        call_mark = len(self.doc.source._calls)
        decoded, missing = [], []
        for tile in needed:
            try:
                raw = self._read_tile(tile)
                samples = _decode_tile(raw, p, self.doc.order)
            except (TiffError, ValueError, zlib.error, struct.error) as exc:
                missing.append(dict(tile, error=str(exc)))
                continue
            decoded.append((tile, samples))
        if missing:
            detail = '; '.join(
                f"tile{t['tile_index']}: {t['error']}" for t in missing)
            raise MissingBlock(
                f'所需{len(missing)}个tile不可用（{detail}），未生成任何填充像素',
                missing)
        for tile, samples in decoded:
            tw = p['tw']
            for gr in range(tile['window_row_start'], tile['window_row_stop']):
                local_row = gr - tile['tile_image_row']
                for gc in range(tile['window_col_start'], tile['window_col_stop']):
                    local_col = gc - tile['tile_image_col']
                    src = (local_row * tw + local_col) * spp
                    dst = ((gr - row_start) * pixel_cols + (gc - col_start)) * spp
                    values[dst:dst + spp] = samples[src:src + spp]
        if any(v is None for v in values):  # 逻辑保护：缺失tile已在上面中止
            raise MissingBlock('窗口存在未采集像素', needed)
        if p['depth'] == 8:
            packed = array('B', values)
        elif array('H').itemsize == 2:
            packed = array('H', values)
        else:
            packed = array('I', values)
        tile_reads = [dict(c) for c in self.doc.source.calls[call_mark:]]
        all_reads = [dict(c) for c in self.doc.source.calls]
        return Capture(
            window={'row_start': row_start, 'row_stop': row_stop,
                    'col_start': col_start, 'col_stop': col_stop},
            rows=pixel_rows, cols=pixel_cols,
            samples_per_pixel=spp, bits_per_sample=list(p['bps']),
            depth=p['depth'], byte_order=self.doc.byte_order,
            values=packed, tiles=needed, plan=self._capture_plan(p, needed),
            geo_tags=self.geo_tags(), nodata=self.nodata_metadata(),
            directory=self.doc.directory_plan(),
            source_calls=all_reads, tile_reads=tile_reads,
            tile_bytes_read=sum(r['length'] for r in tile_reads),
            source_unchanged=True)

    def _capture_plan(self, p, needed):
        return {
            'source': self.doc.source.identify(),
            'directory_digest': self.doc.directory_digest,
            'ifd_index': self.ifd['index'], 'ifd_offset': self.ifd['offset'],
            'byte_order': self.doc.byte_order,
            'tile_offsets': list(p['offsets']),
            'tile_byte_counts': list(p['counts']),
            'tile_grid': {'columns': p['cols'], 'rows': p['rows']},
            'tile_size': {'width': p['tw'], 'height': p['th']},
            'expected_decoded_tile_bytes': p['expected'],
            'compression': p['comp'], 'predictor': p['pred'],
            'planar_configuration': p['plan'],
            'window_to_tiles': [{
                'tile_index': t['tile_index'],
                'tile_row': t['tile_row'], 'tile_col': t['tile_col'],
                'source_range': [t['source_start'],
                                 t['source_start'] + t['source_length']],
                'source_start': t['source_start'],
                'source_length': t['source_length'],
                'window_rows': [t['window_row_start'], t['window_row_stop']],
                'window_cols': [t['window_col_start'], t['window_col_stop']],
                'crop_offset_in_tile': [t['crop_row_offset'],
                                        t['crop_col_offset']],
            } for t in needed],
        }

    def _read_tile(self, tile):
        start, length = tile['source_start'], tile['source_length']
        if length == 0:
            raise TiffError('tile字节数为0')
        if start < 0 or start + length > len(self.doc.source):
            raise TiffError('tile字节区间越界')
        # 缓存键绑定源对象身份与摘要，不同源对象即使字节相同也不共享
        key = (id(self.doc.source), self.doc.source.sha256,
               id(self.doc.source._read_at), start, length)
        if key in self._cache:
            return self._cache[key]
        raw = self.doc.source.read_at(start, length)
        self._cache[key] = raw
        return raw


def _check_window(window, width, height):
    if not isinstance(window, (tuple, list)) or len(window) != 4:
        raise ValueError('窗口必须是(row_start,row_stop,col_start,col_stop)')
    row_start, row_stop, col_start, col_stop = window
    for value in window:
        try:
            _int(value, '窗口坐标')
        except TypeError as exc:
            raise ValueError(str(exc)) from None
    if not 0 <= row_start < row_stop <= height:
        raise ValueError('窗口行范围非法')
    if not 0 <= col_start < col_stop <= width:
        raise ValueError('窗口列范围非法')
    return (row_start, row_stop, col_start, col_stop)


def _make_plan(order, ifd, source_length):
    entries = ifd['entries']
    for tag in (256, 257, 258, 259, 322, 323, 324, 325):
        if tag not in entries:
            if tag in (322, 323, 324, 325):
                raise Unsupported('仅支持tiled图像，strip图像不支持')
            raise TiffError(f'缺少必需tag{tag}')
    if 273 in entries or 279 in entries:
        raise Unsupported('混用strip与tile编码不支持')
    width = _single(order, entries, 256, True)
    length = _single(order, entries, 257, True)
    if not (0 < width <= _MAX_DIMENSION) or not (0 < length <= _MAX_DIMENSION):
        raise TiffError('图像尺寸必须为正且不溢出')
    if width * length > _MAX_IMAGE_PIXELS:
        raise TiffError('图像像素数溢出')
    samples = _single(order, entries, 277)
    spp = 1 if samples is None else samples
    if not 1 <= spp <= 1024:
        raise TiffError('SamplesPerPixel非法')
    bits = _ints(order, entries, 258)
    if len(bits) == 1 and spp > 1:
        raise TiffError('BitsPerSample计数与SamplesPerPixel不符')
    if len(bits) != spp:
        raise TiffError('BitsPerSample计数与SamplesPerPixel不符')
    if any(b not in (8, 16) for b in bits):
        raise Unsupported('仅支持unsigned 8或16位像素')
    if len(set(bits)) != 1:
        raise Unsupported('各波段位深不一致不支持')
    formats = _ints(order, entries, 339)
    if formats is None:
        formats = [1] * spp
    if len(formats) != spp:
        raise TiffError('SampleFormat计数与SamplesPerPixel不符')
    if any(f != 1 for f in formats):
        raise Unsupported('仅支持unsigned采样格式')
    compression = _single(order, entries, 259, True)
    if compression not in (1, 8):
        raise Unsupported(f'Compression={compression}不支持')
    photometric = _single(order, entries, 262)
    planar = _single(order, entries, 284)
    if planar is None:
        planar = 1
    if planar == 2:
        raise Unsupported('PlanarConfiguration=2不支持')
    if planar != 1:
        raise Unsupported(f'PlanarConfiguration={planar}不支持')
    predictor = _single(order, entries, 317)
    if predictor is None:
        predictor = 1
    if predictor not in (1, 2):
        raise Unsupported(f'Predictor={predictor}不支持')
    if predictor == 2 and compression != 8:
        raise Unsupported('Predictor=2仅与Compression=8一起支持')
    tile_w = _single(order, entries, 322, True)
    tile_h = _single(order, entries, 323, True)
    if not (0 < tile_w <= _MAX_DIMENSION) or not (0 < tile_h <= _MAX_DIMENSION):
        raise TiffError('tile尺寸必须为正且不溢出')
    if tile_w * tile_h > _MAX_IMAGE_PIXELS:
        raise TiffError('tile尺寸溢出')
    tile_cols = (width + tile_w - 1) // tile_w
    tile_rows = (length + tile_h - 1) // tile_h
    tile_count = tile_cols * tile_rows
    if tile_count > _MAX_TILES:
        raise TiffError('tile数量溢出')
    expected = tile_w * tile_h * spp * (bits[0] // 8)
    if expected > _MAX_IMAGE_PIXELS:
        raise TiffError('完整tile字节数溢出')
    offsets = _ints(order, entries, 324, tile_count, True)
    counts = _ints(order, entries, 325, tile_count, True)
    for i, (offset, count) in enumerate(zip(offsets, counts)):
        if offset < 0 or count < 0:
            raise TiffError(f'tile{i}偏移或字节数为负')
        if count == 0:
            raise TiffError(f'tile{i}字节数为0')
        if offset < 8 or offset > source_length or \
                count > source_length or offset + count > source_length:
            raise TiffError(f'tile{i}字节区间越界')
        if compression == 1 and count != expected:
            raise TiffError(f'未压缩tile{i}字节数{count}不等于完整tile{expected}')
    return {'width': width, 'length': length, 'spp': spp, 'bps': bits,
            'depth': bits[0], 'comp': compression, 'photo': photometric,
            'plan': planar, 'pred': predictor, 'fmt': formats,
            'tw': tile_w, 'th': tile_h, 'cols': tile_cols, 'rows': tile_rows,
            'offsets': offsets, 'counts': counts, 'expected': expected}


def _decode_tile(raw, plan, order):
    expected = plan['expected']
    if plan['comp'] == 1:
        if len(raw) != expected:
            raise TiffError(f'未压缩tile字节数{len(raw)}不等于预期{expected}')
        data = raw
    else:
        if not raw:
            raise TiffError('空Deflate流')
        decoder = zlib.decompressobj()
        try:
            data = decoder.decompress(raw, expected + 1)
        except zlib.error as exc:
            raise TiffError(f'Deflate解压失败: {exc}') from exc
        if len(data) > expected:
            raise TiffError(f'Deflate解压输出超过预期{expected}字节')
        if decoder.unconsumed_tail:
            raise TiffError(f'Deflate解压输出超过预期{expected}字节')
        if not decoder.eof:
            raise TiffError('Deflate流不完整')
        if decoder.unused_data:
            raise TiffError('Deflate流含尾随数据')
        try:
            tail = decoder.flush()
        except zlib.error as exc:
            raise TiffError(f'Deflate刷新失败: {exc}') from exc
        if len(data) + len(tail) != expected:
            raise TiffError(
                f'Deflate解压输出{len(data) + len(tail)}不等于预期{expected}字节')
        data = data + tail
    samples = _bytes_to_samples(data, plan, order)
    if plan['pred'] == 2:
        samples = _undo_horizontal_predictor(samples, plan)
    return samples


def _bytes_to_samples(data, plan, order):
    total = plan['tw'] * plan['th'] * plan['spp']
    if plan['depth'] == 8:
        if len(data) != total:
            raise TiffError('8位tile字节数与完整tile不符')
        return list(data)
    if len(data) != total * 2:
        raise TiffError('16位tile字节数与完整tile不符')
    # 16位加法必须先按文件字节序解析成整数，再恢复预测值
    return list(struct.unpack(order + 'H' * total, data))


def _undo_horizontal_predictor(values, plan):
    """逐tile、逐行、逐波段恢复；每行重置，不把末值带到下一行。"""
    spp = plan['spp']
    row_len = plan['tw'] * spp
    mask = 0xff if plan['depth'] == 8 else 0xffff
    for row in range(plan['th']):
        base = row * row_len
        for i in range(base + spp, base + row_len):
            values[i] = (values[i] + values[i - spp]) & mask
    return values


@dataclass
class Capture:
    """一个窗口的采集结果；所有坐标均为原图行列。"""
    window: dict
    rows: int
    cols: int
    samples_per_pixel: int
    bits_per_sample: list
    depth: int
    byte_order: str
    values: array
    tiles: list
    plan: dict
    geo_tags: dict
    nodata: dict
    directory: dict
    source_calls: list
    tile_reads: list
    tile_bytes_read: int
    source_unchanged: bool

    def pixel(self, row, col):
        """按原图坐标取一个像素的全部波段值。"""
        if not (self.window['row_start'] <= row < self.window['row_stop'] and
                self.window['col_start'] <= col < self.window['col_stop']):
            raise ValueError('像素不在采集窗口内')
        base = ((row - self.window['row_start']) * self.cols +
                (col - self.window['col_start'])) * self.samples_per_pixel
        return list(self.values[base:base + self.samples_per_pixel])

    def band(self, row, col, band=0):
        if not 0 <= band < self.samples_per_pixel:
            raise ValueError('波段序号超出范围')
        return self.pixel(row, col)[band]

    def pixel_array(self):
        """窗口内按行优先的紧凑像素列表，每像素spp个值。"""
        return [list(self.values[i:i + self.samples_per_pixel])
                for i in range(0, len(self.values), self.samples_per_pixel)]

    def tile_stats(self):
        return [{'tile_index': t['tile_index'],
                 'tile_row': t['tile_row'], 'tile_col': t['tile_col'],
                 'source_start': t['source_start'],
                 'source_length': t['source_length'],
                 'source_range': [t['source_start'],
                                  t['source_start'] + t['source_length']],
                 'window_rows': [t['window_row_start'],
                                 t['window_row_stop']],
                 'window_cols': [t['window_col_start'],
                                 t['window_col_stop']],
                 'crop_offset_in_tile': [t['crop_row_offset'],
                                         t['crop_col_offset']]}
                for t in self.tiles]

    def to_dict(self):
        return {
            'source': self.directory['source'],
            'directory_digest': self.directory['directory_digest'],
            'directory_ranges': self.directory['directory_ranges'],
            'ifd_index': self.plan['ifd_index'],
            'window': dict(self.window),
            'rows': self.rows, 'cols': self.cols,
            'samples_per_pixel': self.samples_per_pixel,
            'bits_per_sample': list(self.bits_per_sample),
            'byte_order': self.byte_order,
            'pixels': self.pixel_array(),
            'tiles': self.tile_stats(),
            'tile_grid': self.plan['tile_grid'],
            'tile_size': self.plan['tile_size'],
            'tile_offsets': self.plan['tile_offsets'],
            'tile_byte_counts': self.plan['tile_byte_counts'],
            'expected_decoded_tile_bytes': self.plan[
                'expected_decoded_tile_bytes'],
            'window_to_tiles': self.plan['window_to_tiles'],
            'compression': self.plan['compression'],
            'predictor': self.plan['predictor'],
            'planar_configuration': self.plan['planar_configuration'],
            'geo_tags': self.geo_tags,
            'nodata': self.nodata,
            'tile_reads': self.tile_reads,
            'tile_bytes_read': self.tile_bytes_read,
            'source_unchanged': self.source_unchanged,
        }
