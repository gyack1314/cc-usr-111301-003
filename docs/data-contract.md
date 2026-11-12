# TIFF窗口资料

窗口均为(row_start,row_stop,col_start,col_stop)半开整数，按原图像素行列定义，不做地图投影重采样。主图3列2行，每像素两通道unsigned16，文件大端，tile宽高各2。ExtraSamples=0代表第二通道未解释，所有通道原值均采集。tile边缘填充存在于压缩块中，不属于图像窗口；采集时完整解码tile，再按图像边界与窗口裁切。

## 目录

- 经典TIFF：字节序标记II或MM，magic=42；BigTIFF（magic=43）明确拒绝。
- IFD链可含多个独立图像。调用者必须给明确`ifd_index`选图；`Doc.images()`只列摘要，缩略图与主图同样列出，工具不会默认选首个IFD。
- 字段类型支持BYTE/ASCII/SHORT/LONG/RATIONAL/SBYTE/SSHORT/SLONG/FLOAT/DOUBLE/SRATIONAL的目录取值，未解释tag保留原值、类型、计数、字节长度和读取位置（内联或偏移）。关键像素布局tag（ImageWidth/Length、BitsPerSample、Compression、Photometric、SamplesPerPixel、PlanarConfiguration、Predictor、TileWidth/Length、TileOffsets/Counts、SampleFormat）重复即拒绝。
- IFD链出现环、偏移越界、值偏移越界或长度不符、尺寸/计数/tile数溢出，均拒绝目录；目录解析阶段不读取像素字节，也不会把tile偏移当作像素位置猜测。

## 选定图像

仅支持：tiled；单波段或chunky多波段（PlanarConfiguration=1，缺省视为1）；unsigned 8/16位（SampleFormat=1）；Compression=1（未压缩）或8（zlib封装Deflate，不是raw Deflate）。Predictor=1或2，Predictor=2只与Compression=8配合。PlanarConfiguration=2、strip图像、LZW/JPEG等其他压缩、有符号采样、混合strip与tile均明确`Unsupported`。

TileOffsets/TileByteCounts按行优先tile网格索引（`tile_index = tile_row * tile_cols + tile_col`），数量必须等于网格tile数；源区间在选图时即校验。

## 解码

- 每个tile按完整tile尺寸解码，窗口只取相交区域；边缘块保留padding但不进入窗口。
- Deflate只接受解压后精确预期字节数（tile_width*tile_height*samples*bytes_per_sample）。超额输出、短输出、流不完整或尾随数据均失败。
- 水平预测器按每个tile、每一行、每一波段的模2位数幂恢复；每行起点重置，不把前一行末值带到下一行。16位先按文件字节序解析成整数再做加法。
- 所需tile读取或解码失败时抛`MissingBlock`，携带失败tile的索引与源字节区间；不返回任何填零后的完整图像，源对象保持不变。

## 坐标与结果

结果`Capture`保留原图坐标：`window`、`rows/cols`、紧凑行优先像素数组（每像素samples_per_pixel个值）、`pixel(row,col)`按原图行列取回、`tile_stats()`给出窗口到tile及源字节区间的映射、`window_to_tiles`在计划中重复该映射。`geo_tags()`返回GeoKeyDirectory、ModelPixelScale、ModelTiepoint、ModelTransformation、GeoDoubleParams、GeoAsciiParams的原值，不做坐标换算。

NoData（GDAL_NODATA，tag 42113）只作为带`source`的元数据返回，`is_fill_value`恒为false；绝不用它填充未下载区域。

## 源与计划

`ImageSource`绑定长度与对象摘要。`read_at(start,length)`核验返回类型与真实长度，记录实际区间及其字节摘要；越界、短读、非法参数均失败。计划（`Doc.directory_plan()`与`Capture.plan`）绑定源sha256/长度、目录摘要和全部目录读取区间；缓存键包含源对象身份、摘要和底层读取函数身份，不能跨对象身份共享。

read_at允许合并相邻范围但不得借一次整对象读取掩盖tile粒度效果。源长度和sha来自可信对象清单，生产接入中须核对返回Content-Range与对象身份；本地冻结资产只做同等read_at。初始`capture_data.io.Source`不解析TIFF，坏块是域操作时应发现的源错误，材料本身sha正确。`python -m capture_data.cli capture <manifest.json>`输出像素、tile映射与元数据；缺块时输出`missing_blocks`及缺失tile区间。
