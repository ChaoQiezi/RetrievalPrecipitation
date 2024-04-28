# @Author   : ChaoQiezi
# @Time     : 2024/4/26  17:16
# @FileName : dead_code2.py
# @Email    : chaoqiezi.one@qq.com

"""
This script is used to ...
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': '2m_dewpoint_temperature',
        'year': '1951',
        'area': [
            90, -180, -90,
            180,
        ],
        'month': '02',
        'day': '03',
        'time': '02:00',
        'format': 'netcdf.zip',
    },
    'download.netcdf.zip')