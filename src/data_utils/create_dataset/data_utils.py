import dbutils
import pandas as pd
import logging

def extract_qrcode(row):
    """
    get the qrcode from the artifacts

    Args:
        row (dataframe rows): complete row of a dataframe

    Returns:
        string: qrcodes extracted from the artifacts
    """
    qrc = row['storage_path']
    split_qrc = qrc.split('/')[1]
    return split_qrc


class CollectQrcodes:
    """
    class to gather a qrcodes from backend to prepare dataset for model training and evaluation.

    """

    def __init__(self, db_connector):
        """
        Args:
            db_connector (json_file): json file to connect to the database.
        """
        self.db_connector = db_connector
        self.ml_connector = dbutils.connect_to_main_database(self.db_connector)

    def get_all_data(self):
        """
        Gather the qrcodes from the databse with labels.

        Returns:
            dataframe: panda dataframe contains data from databse.
        """
        table_name = 'artifacts_with_target'
        columns = self.ml_connector.get_columns(table_name)
        query = "select * from " + table_name + " where id like '%_version_5.0%';"
        database = self.ml_connector.execute(query, fetch_all=True)
        database = pd.DataFrame(database, columns=columns)
        database['qrcode'] = database.apply(extract_qrcode, axis=1)
        return database

    def get_scangroup_data(self, data, scangroup):
        """
        Get the data from the available pool and scan_group pool

        Args:
            data (dataframe): dataframe having all the data from the database
            scangroup (string): 'train or 'test' scan_group

        Returns:
            dataframe: dataframe having data from the specific scan_group type or null type.
        """
        scangroup_data = data[(data['scan_group'] == scangroup) | (data['scan_group'].isnull())]
        return scangroup_data

    def get_unique_qrcode(self, dataframe):
        """
        Get a unique qrcodes from the dataframe file and return dataframe with qrcodes, scan_group, tags.

        Args:
            dataframe (panda dataframe): A panda dataframe file with qrcodes, artifacts, and all the metadata.
        """
        data = dataframe.drop_duplicates(subset=["qrcode"], keep='first')
        unique_qrcode_data = data[['qrcode', 'scan_group', 'tag']]

        return unique_qrcode_data

    def get_usable_data(self, dataframe, amount, scan_group='train'):
        """[summary]

        Args:
            dataframe ([type]): [description]
            amount ([type]): [description]
        """
        available_data = dataframe[dataframe['scan_group'].isnull()]
        used_data = dataframe[dataframe['scan_group'] == scan_group]
        required_amount = int(amount) - len(used_data)
        if required_amount <= 0:
            return logging.warning("Amount scans given is less than already used scans")

        remain_data = available_data.sample(n=amount - len(used_data))
        dataList = [used_data, remain_data]
        complete_data = pd.concat(dataList)
        return complete_data

    def merge_qrcode_dataset(self, qrcodes, dataset):
        """
        Merge the qrcodes dataframe  and whole database dataframe
        Args:
            qrcodes (dataframe): dataframe containing unique qrcodes
            dataset (dataframe): dataframe containing all the data from the database
        """
        qrcodes = qrcodes['qrcode']
        full_dataset = pd.merge(qrcodes, dataset, on='qrcode', how='left')
        return full_dataset

    def get_posenet_results(self):
        """
        Fetch the posenet data for RGB and collect their ids.

        """
        artifact_result = "select * from artifact_result where artifact_id like '%_version_5.0%' and model_id ='posenet_1.0';"
        artifacts_columns = self.ml_connector.get_columns('artifact_result')
        artifacts_table = self.ml_connector.execute(artifact_result, fetch_all=True)
        artifacts_frame = pd.DataFrame(artifacts_table, columns=artifacts_columns)
        artifacts_frame = artifacts_frame.rename(columns={"artifact_id": "id"})
        return artifacts_frame

    def get_artifacts(self):
        """
        Get the artifacts results from the database for RGB
        """

        query = "select id,storage_path,qr_code from artifact where id like '%_version_5.0%' and dataformat ='rgb';"
        artifacts = self.ml_connector.execute(query, fetch_all=True)
        artifacts = pd.DataFrame(artifacts, columns=['id', 'storage_path', 'qrcode'])
        return artifacts

    def merge_data_artifacts(self, data, artifacts):
        """
        Merge the two dataset of artifacts and posenet database

        Args:
            data (dataframe):  dataframe with  qrcodes and all the other labels.
            artifacts (dataframe): artifacts.
        """

        rgb_data = pd.merge(data[['qrcode']], artifacts, on='qrcode', how='left')
        results = rgb_data.drop_duplicates(subset='storage_path', keep='first', inplace=False)
        results = results.drop_duplicates(subset='id', keep='first', inplace=False)
        return results

    def merge_data_posenet(self, data, posenet):
        """
        Merge the dataset and posenet datafarme to gather posenent results for available RGB

        Args:
            data (datafarme): dataframe with artifacts data
            posenet (dataframe): posenet data with keypoints for different bodypart.
        """
        posenet_results = pd.merge(data, posenet[['id', 'json_value', 'confidence_value']], on='id', how='left')
        return posenet_results

    def update_database(self, data, scan_group):
        """
        Update the dataabse with the qrcodes with assigned scan_group

        Args:
            data (dataframe): Dataframe containing the qrcodes  used for dataset preparation.
            scan_group(string): scan_group containing two option train and test.

        Returns:
            Database being updates with the scan_group
        """
        final_training_qrcodes = []
        ignored_training_qrcodes = []
        scan_group_qrcode = data['qrcode'].values.tolist()
        for qrcode in scan_group_qrcode:
            select_statement = "select id from measure where type like 'v%' and id like '%version_5.0%' and qr_code = '{}';".format(
                qrcode)
            id = self.ml_connector.execute(select_statement, fetch_all=True)
            if len(id) == 1:
                final_training_qrcodes.append(qrcode)
            else:
                ignored_training_qrcodes.append(qrcode)

        for id in final_training_qrcodes:
            update_statement = "update measure set scan_group ='{}' where qr_code = '{}';".format(scan_group, id)
            try:
                self.ml_connector.execute(update_statement)
            except Exception as error:
                logging.warning(error)
        return
