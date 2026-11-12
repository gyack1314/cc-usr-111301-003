"""测试专用的经典TIFF构造器；只在本目录被测试导入。"""
import struct
import zlib

_TYPE_FMT = {1: 'B', 2: 's', 3: 'H', 4: 'I', 5: 'I', 6: 'b',
             7: 'B', 8: 'h', 9: 'i', 10: 'i', 11: 'f', 12: 'd'}


def pack_values(order, typ, cnt, values):
    """按TIFF字段类型打包值；bytes原样返回。"""
    if isinstance(values, (bytes, bytearray)):
        return bytes(values)
    if typ == 2:
        if isinstance(values, str):
            values = values.encode('latin-1')
        return bytes(values)
    if typ in (5, 10):
        flat = []
        for pair in values:
            flat.extend(pair)
        return struct.pack(order + _TYPE_FMT[typ] * len(flat), *flat)
    fmt = _TYPE_FMT[typ]
    return struct.pack(order + fmt * cnt, *values)


def build_tiff(order, ifds, tail=b''):
    """写经典TIFF。

    order为'<'或'>'。ifds是字典列表：{'tags':[(tag,type,count,values),...],
    'next': 可选显式下一IFD偏移（测试环/越界用）}。
    第0个IFD偏移8；其余IFD紧跟其后；随后是外部值块，最后是tail。
    """
    marker = b'II' if order == '<' else b'MM'
    encoded = []
    for ifd in ifds:
        rows = []
        for tag, typ, cnt, values in ifd['tags']:
            stored = pack_values(order, typ, cnt, values)
            rows.append((tag, typ, cnt, stored))
        encoded.append(rows)
    ifd_offset = []
    cursor = 8
    for rows in encoded:
        ifd_offset.append(cursor)
        cursor += 2 + 12 * len(rows) + 4
    externals = []  # (ifd可以共享同样值块不必要；逐个分配)
    for rows in encoded:
        for tag, typ, cnt, stored in rows:
            if len(stored) > 4:
                cursor = (cursor + 1) & ~1
                externals.append((tag, cnt, stored, cursor))
                cursor += len(stored)
    tail_offset = (cursor + 1) & ~1
    out = bytearray(tail_offset + len(tail))
    out[0:8] = struct.pack(order + '2sHI', marker, 42, ifd_offset[0])
    for index, rows in enumerate(encoded):
        pos = ifd_offset[index]
        struct.pack_into(order + 'H', out, pos, len(rows))
        pos += 2
        used = {}
        for tag, typ, cnt, stored in rows:
            struct.pack_into(order + 'HHI', out, pos, tag, typ, cnt)
            if len(stored) <= 4:
                out[pos + 8:pos + 12] = stored.ljust(4, b'\0')
            else:
                key = (tag, cnt, stored)
                if key not in used:
                    for t2, c2, s2, off in externals:
                        if (t2, c2, s2) == key:
                            used[key] = off
                            out[off:off + len(s2)] = s2
                            break
                struct.pack_into(order + 'I', out, pos + 8, used[key])
            pos += 12
        nxt = ifds[index].get('next')
        if nxt is None:
            nxt = ifd_offset[index + 1] if index + 1 < len(ifds) else 0
        struct.pack_into(order + 'I', out, pos, nxt)
    out[tail_offset:tail_offset + len(tail)] = tail
    return bytes(out), tail_offset


def tile_bytes(width, height, tile_w, tile_h, spp=1, depth=16,
               compression=8, predictor=1, order='>', seed=0, edge_pad=7):
    """生成按行优先tile网格排列的像素与压缩字节。

    返回(offsets, counts, payloads, pixels)。边缘padding使用edge_pad值，
    以便测试是否错误地把padding当作图像内容。
    """
    mod = 65536 if depth == 16 else 256
    cols = (width + tile_w - 1) // tile_w
    rows = (height + tile_h - 1) // tile_h
    pixels = [[(1 + seed + r * 97 + c * spp + s) % mod for s in range(spp)]
              for r in range(height) for c in range(width)]
    payloads = []
    for tr in range(rows):
        for tc in range(cols):
            vals = []
            for lr in range(tile_h):
                for lc in range(tile_w):
                    gr, gc = tr * tile_h + lr, tc * tile_w + lc
                    if gr < height and gc < width:
                        vals.extend(pixels[gr * width + gc])
                    else:
                        vals.extend([edge_pad % mod] * spp)
            if predictor == 2:
                mask = 0xff if depth == 8 else 0xffff
                row_len = tile_w * spp
                for row in range(tile_h):
                    base = row * row_len
                    for i in range(base + row_len - 1, base + spp - 1, -1):
                        vals[i] = (vals[i] - vals[i - spp]) & mask
            raw = bytes(vals) if depth == 8 else struct.pack(order + 'H' * len(vals), *vals)
            payloads.append(zlib.compress(raw, 9) if compression == 8 else raw)
    return payloads, pixels


def tags_for(width, height, tile_w, tile_h, spp=1, depth=16, compression=8,
             predictor=None, planar=None, photo=1, sample_format=(1,),
             offsets=None, counts=None, nodata=None, geo=True, subfile=None):
    tags = [(256, 4, 1, [width]), (257, 4, 1, [height]),
            (258, 3, spp, [depth] * spp), (259, 3, 1, [compression]),
            (262, 3, 1, [photo])]
    if spp != 1:
        tags.append((277, 3, 1, [spp]))
    if planar is not None:
        tags.append((284, 3, 1, [planar]))
    if predictor is not None:
        tags.append((317, 3, 1, [predictor]))
    tags += [(322, 4, 1, [tile_w]), (323, 4, 1, [tile_h]),
             (324, 4, len(offsets), list(offsets)),
             (325, 4, len(counts), list(counts))]
    if sample_format:
        tags.append((339, 3, len(sample_format), list(sample_format)))
    if subfile is not None:
        tags.append((254, 4, 1, [subfile]))
    if nodata is not None:
        tags.append((42113, 2, len(nodata) + 1, nodata))
    if geo:
        tags += [(33550, 12, 3, [1.0, 1.0, 0.0]),
                 (33922, 12, 6, [0.0, 0.0, 0.0, 10.0, 20.0, 0.0]),
                 (34735, 3, 8, [1, 1, 0, 1, 1024, 0, 1, 1]),
                 (34737, 2, 4, 'UTM\x00')]
    return tags


def _external_offsets(order, tags, tail):
    """一次构建，取得tag值块之后的payload起始偏移；仅用于单IFD构造。"""
    blob, _ = build_tiff(order, [{'tags': tags}], b'')
    return len(blob)


def assemble(order, images):
    """按顺序把若干图像的payload放在全部目录之后。

    images: [(tags_factory, payloads)]，tags_factory(offsets, counts)返回tag列表。
    返回(blob, [offsets...])。
    """
    probe_tags = [factory([0] * len(payloads), [len(p) for p in payloads])
                  for factory, payloads in images]
    blob, base = build_tiff(order, [{'tags': t} for t in probe_tags], b'')
    cursor = base
    results = []
    for _, payloads in images:
        offsets = []
        for payload in payloads:
            offsets.append(cursor)
            cursor += len(payload)
        results.append(offsets)
    tags = [factory(offsets, [len(p) for p in payloads])
            for (factory, payloads), offsets in zip(images, results)]
    joined = b''.join(b''.join(payloads) for _, payloads in images)
    return build_tiff(order, [{'tags': t} for t in tags], joined)[0], results


def _image_factory(width, height, tile_w, tile_h, spp, depth, compression,
                   predictor, planar, photo, sample_format, nodata,
                   extra_tags=(), geo=False):
    def factory(offsets, counts):
        tags = tags_for(width, height, tile_w, tile_h, spp, depth, compression,
                        predictor, planar, photo, sample_format, offsets,
                        counts, nodata, geo=geo)
        return tags + list(extra_tags)
    return factory


def single_image(width, height, tile_w, tile_h, order='>', spp=1, depth=16,
                 compression=8, predictor=None, planar=None,
                 sample_format=None, photo=1, nodata=None, seed=0,
                 edge_pad=7, extra_tags=(), geo=False):
    payloads, pixels = tile_bytes(width, height, tile_w, tile_h, spp, depth,
                                  compression, predictor or 1, order, seed,
                                  edge_pad)
    if sample_format is None:
        sample_format = (1,) * spp
    factory = _image_factory(width, height, tile_w, tile_h, spp, depth,
                             compression, predictor, planar, photo,
                             sample_format, nodata, extra_tags, geo)
    blob, offsets = assemble(order, [(factory, payloads)])
    return blob, pixels, offsets[0], [len(p) for p in payloads]


def multi_image(order, images):
    """images: [{'payloads':..., 'width':..., ...}]；返回(blob, offsets列表)。"""
    factories = []
    for spec in images:
        factory = _image_factory(spec['width'], spec['height'],
                                 spec['tile_w'], spec['tile_h'],
                                 spec.get('spp', 1), spec.get('depth', 16),
                                 spec.get('compression', 8),
                                 spec.get('predictor'), spec.get('planar'),
                                 spec.get('photo', 1),
                                 spec.get('sample_format') or (1,) *
                                 spec.get('spp', 1),
                                 spec.get('nodata'),
                                 spec.get('extra_tags', ()),
                                 spec.get('geo', False))
        factories.append((factory, spec['payloads']))
    return assemble(order, factories)


def one_tile_blob(order, payload, width=2, height=2, tile_w=2, tile_h=2,
                  spp=1, depth=16, compression=8, predictor=None):
    """用任意压缩字节构造单tile图像，用于坏流测试。"""
    counts = [len(payload)]
    tags = [(256, 4, 1, [width]), (257, 4, 1, [height]),
            (258, 3, spp, [depth] * spp), (259, 3, 1, [compression]),
            (262, 3, 1, [1])]
    if spp != 1:
        tags.append((277, 3, 1, [spp]))
    if predictor is not None:
        tags.append((317, 3, 1, [predictor]))
    tags += [(322, 4, 1, [tile_w]), (323, 4, 1, [tile_h]),
             (324, 4, 1, [0]), (325, 4, 1, counts)]
    blob, _ = build_tiff(order, [{'tags': tags}], payload)
    first = len(blob) - len(payload)
    tags[-2] = (324, 4, 1, [first])
    return build_tiff(order, [{'tags': tags}], payload)[0]
