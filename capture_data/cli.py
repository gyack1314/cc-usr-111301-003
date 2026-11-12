"""输出原始清单概况，或按清单采集GeoTIFF窗口。"""
import argparse,json
from pathlib import Path
from .io import load,Source
from .domain import summarize
from .tiff import (ImageSource,MissingBlock,open_tiff)

def _window(v):
    w=v['window']
    return (w['row_start'],w['row_stop'],w['col_start'],w['col_stop'])

def capture(path):
    v,digest=load(path);root=Path(path).parent
    src=Source(root,v['source'])
    doc=open_tiff(ImageSource.from_bytes(src.raw,src.sha256))
    image=doc.select(v['ifd_index'])
    try:
        result=image.capture(_window(v),v.get('max_decoded_tile_bytes',1<<24),
                             v.get('read_budget',1<<28))
    except MissingBlock as exc:
        return {'manifest_sha256':digest,'ifd_index':v['ifd_index'],
                'source':{'path':v['source']['path'],'sha256':src.sha256,'bytes':len(src.raw)},
                'window':v['window'],'status':'missing_blocks',
                'missing':[{'tile_index':t['tile_index'],'source_start':t['source_start'],
                            'source_length':t['source_length'],'error':t['error']} for t in exc.tiles],
                'source_unchanged':True}
    return {'manifest_sha256':digest,'status':'ok','ifd_index':result.plan['ifd_index'],
            'source':{'path':v['source']['path'],'sha256':result.plan['source']['sha256'],
                      'bytes':result.plan['source']['length']},
            'window':result.window,'rows':result.rows,'cols':result.cols,
            'pixels':result.pixel_array(),'tiles':result.tile_stats(),
            'tile_reads':result.tile_reads,'tile_bytes_read':result.tile_bytes_read,
            'geo_tags':result.geo_tags,'nodata':result.nodata,
            'directory_digest':result.plan['directory_digest']}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','capture']);p.add_argument('file');a=p.parse_args()
    if a.command=='inspect':
        v,digest=load(a.file);print(json.dumps({'manifest_sha256':digest,'summary':summarize(v,Path(a.file).parent)},ensure_ascii=False,indent=2))
    else:print(json.dumps(capture(a.file),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
