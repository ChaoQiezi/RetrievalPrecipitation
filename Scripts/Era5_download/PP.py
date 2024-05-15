import cdsapi

# c = cdsapi.Client()
c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2", key="270739:2beffef8-1718-46eb-b534-84ebea6ab0d9")
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': ['2m_dewpoint_temperature',
            '2m_temperature', 'skin_temperature', ],
        'year': ['2010'],
        'month': ['09', '10', '11', '12'],
        'day': ['01', '28'],
        'time': [
            '00:00', '01:00', '02:00','03:00', '04:00', '05:00', '06:00', '07:00', '08:00','09:00', '10:00', '11:00', '12:00', '13:00','14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'area': [
            26, 73, 40,
            105,
        ],
        'format': 'netcdf.zip'
    },
    r'H:\Datasets\Objects\Temp')

