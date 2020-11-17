import unittest
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from data_utils import CollectQrcodes  # noqa: E402

parameters = 'src/data_utils/create_dataset/parameters.yml'
dbconnection = 'src/data_utils/create_dataset/dbconnection.json'

with open(parameters, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

dataset = CollectQrcodes(dbconnection)

QR_CODE_1 = "1597619052-n0p0wb3o77"
QR_CODE_2 = "1555555555-yqqqqqqqqq"
QR_CODE_3 = "1555836480-mvcbmnfbbb"

data = {
    'id': [f'Uyx4dG24QFMtAk4v_artifact-scan-pcd_{QR_CODE_1}_version_5.0',
           f'Uyx4dGdcsadawAw_artifact-scan-pcd_{QR_CODE_1}_version_5.0',
           f'JsbdjObdasghncz_artifact-scan-pcd_{QR_CODE_2}_version_5.0',
           f'JsbdjObdakcxvkj_artifact-scan-pcd_{QR_CODE_3}_version_5.0'],
    'storage_path': [
        f'qrcode/{QR_CODE_1}/measure/1591468683124/pc/pc_{QR_CODE_1}_1591468683124_100_001.pcd',
        f'qrcode/{QR_CODE_1}/measure/1591468683124/pc/pc_{QR_CODE_1}_1591468683124_101_001.pcd',
        f'qrcode/{QR_CODE_2}/measure/1591468683124/pc/pc_{QR_CODE_2}_1591468683124_102_001.pcd',
        f'qrcode/{QR_CODE_3}/measure/1591468683124/pc/pc_{QR_CODE_3}_1591468683124_201_001.pcd'],
    'height': [90.9, 90.9, 92.1, 100.2],
    'weight': [15.3, 15.3, 11.3, 18.5],
    'muac': [13.2, 13.2, 16, 15.3],
    'scan_group': ['train', 'train', 'test', np.nan],
    'key': [100, 101, 102, 201],
    'tag': ['good', 'bad', 'good', 'bad'],
    'age': [1223, 1223, 1645, 987],
    'sex': ['male', 'male', 'female', 'female'],
    'qrcode': [QR_CODE_1, QR_CODE_1, QR_CODE_2, QR_CODE_3]

}
dataframe = pd.DataFrame.from_dict(data)
expected_data = dataset.get_all_data()
expected_scangroup_data = dataset.get_scangroup_data(data=dataframe, scangroup='test')
unique_qrcodes = dataset.get_unique_qrcode(dataframe)


class TestListElements(unittest.TestCase):

    def test_dataset_column(self):
        """
        Function to test required columns match with columns from database.
        """
        expected = list(dataframe.columns)
        result = list(expected_data.columns)
        self.assertCountEqual(result, expected)

    def test_scan_group(self):
        """
        Function to test the scan_group fetch from database
        """
        result = list(expected_scangroup_data['scan_group'])
        assert 'train' not in result

    def test_unique_qrcode(self):
        """
        Function to fetch the unique qrcodes from the dataframe
        """
        qrcode = unique_qrcodes['qrcode'].tolist()
        assert np.unique(qrcode).size == len(qrcode)


if __name__ == "__main__":
    unittest.main()
