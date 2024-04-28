# @Author   : ChaoQiezi
# @Time     : 2024/3/4  23:06
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import cdsapi


c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2", key="291992:1ce2a83d-6e78-459d-a093-e3d66cf8b6f0")
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': '2m_dewpoint_temperature',
        'format': 'netcdf.zip',
        'year': ['2020', '2023'],
        'month': '01',
        'day': '01',
        'time': [
            '00:00'
        ],
    },
    'download.netcdf.zip')


import numpy as np
from typing import Union
from osgeo import gdal, osr, ogr

def img_warp(src_img: np.ndarray, out_path: str, transform: list,
             nodata: Union[int, float] = None) -> None:
    """
    该函数用于对正弦投影下的栅格矩阵进行重投影(GLT校正), 得到WGS84坐标系下的栅格矩阵并输出为TIFF文件
    :param src_img: 待重投影的栅格矩阵
    :param out_path: 输出路径
    :param transform: 仿射变换参数([x_min, x_res, 0, y_max, 0, -y_res], 旋转参数为0是常规选项)
    :param out_res: 输出的分辨率(栅格方形)
    :param nodata: 设置为NoData的数值
    :param out_type: 输出的数据类型
    :param resample: 重采样方法(默认是最近邻, ['nearest', 'bilinear', 'cubic'])
    :param src_proj4: 表达源数据集(src_img)的坐标系参数(以proj4字符串形式)
    :return: None
    """

    # 原始数据集创建
    driver = gdal.GetDriverByName('GTiff')  # 在内存中临时创建
    src_ds = driver.Create(out_path, src_img.shape[1], src_img.shape[0], 1, gdal.GDT_Float32)  # 注意: 先传列数再传行数, 1表示单波段
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    src_ds.SetProjection(srs.ExportToWkt())  # 设置投影信息
    src_ds.SetGeoTransform(transform)  # 设置仿射参数
    src_ds.GetRasterBand(1).WriteArray(src_img)  # 写入数据
    src_ds.GetRasterBand(1).SetNoDataValue(nodata)
