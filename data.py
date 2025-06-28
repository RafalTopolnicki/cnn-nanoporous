import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STRUCTURE_ID = "structure_id"


def get_npy_shape(filepath: str):
    dat = np.load(filepath)
    return dat.shape


class DescDimReducer:
    def __init__(self, df: pd.DataFrame, pca_n_components=20, scale_only=False):
        self.database = df
        self.scale_only = scale_only
        self.descriptors_columns = [x for x in self.database.columns if x not in ["npy_path", "cii", "direction"]]
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.descriptors_columns])
        if not scale_only:
            scaled_data = self.scaler.transform(df[self.descriptors_columns])
            self.pca = PCA(n_components=pca_n_components)
            self.pca.fit(scaled_data)

    def transform(self, df):
        if len(df) == 0:
            return df
        data = df[self.descriptors_columns]
        df_other = df.drop(self.descriptors_columns, axis=1)
        scaled_data = self.scaler.transform(data)
        if self.scale_only:
            df_scaled = pd.DataFrame(scaled_data)
            df_scaled = pd.concat([df_other, df_scaled.set_index(df_other.index)], axis=1)
            return df_scaled
        transformed = self.pca.transform(scaled_data)
        df_transformed = pd.DataFrame(transformed)
        column_names = [f"pca_{i}" for i in range(self.pca.n_components)]
        df_transformed.columns = column_names
        df_transformed = pd.concat([df_other, df_transformed.set_index(df_other.index)], axis=1)
        return df_transformed


class MDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        use_descriptors: bool = False,
        descriptor_transform: DescDimReducer = None,
    ):
        self.database = df
        self.transform = transform
        self.use_descriptors = use_descriptors
        self.descriptors_columns = None
        if use_descriptors:
            self.descriptors_columns = [x for x in self.database.columns if x not in ["npy_path", "cii"]]
            if descriptor_transform is not None:
                self.database = descriptor_transform.transform(self.database)
                self.descriptors_columns = [x for x in self.database.columns if x not in ["npy_path", "cii"]]
            print(self.descriptors_columns)
            print(len(self.descriptors_columns))

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = self.database.iloc[idx]["cii"].astype("float32")
        path = self.database.iloc[idx]["npy_path"]
        structure = np.load(path).astype("float32")
        if self.transform:
            structure = self.transform(structure)
        else:
            structure = np.expand_dims(structure, axis=0)
        if self.use_descriptors:
            descriptors = np.array(self.database[self.descriptors_columns].iloc[idx]).astype("float32")
            return structure, target, path, descriptors
        else:
            return structure, target, path


def get_structure_id_from_filepath(filepath):
    # example for /home/user/cnn-nanoporous/structures/1469.dx_rot-x.npy
    # structure_id is 1469
    # example for /home/user/cnn-nanoporous/structures/3806_rot-x.npy
    # structure_id is 3806
    return int(filepath.replace(".dx", "").split("/")[-1].split("_rot")[0])


def split_into_folds_nostratification(df: pd.DataFrame, cv_folds: int, which_folds: list[int]):
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be at least 1. {cv_folds} provided")
    if cv_folds == 1:
        dftrainval, df_test = train_test_split(df, test_size=0.2)
        df_train, df_val = train_test_split(dftrainval, test_size=0.2)
        yield df_train, df_val, df_test, 0
    else:
        kf = KFold(n_splits=cv_folds)
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(df)):
            if fold_id in which_folds:
                train_idx, val_idx = train_test_split(trainval_idx, test_size=(1 - 0.2) * 0.2)
                assert len(set(train_idx).intersection(val_idx)) == 0, "Overlap between train and val"
                assert len(set(train_idx).intersection(test_idx)) == 0, "Overlap between train and test"
                assert len(set(val_idx).intersection(test_idx)) == 0, "Overlap between val and test"
                df_train = df.iloc[train_idx, :]
                df_val = df.iloc[val_idx, :]
                df_test = df.iloc[test_idx, :]
                yield df_train, df_val, df_test, fold_id


def split_into_folds_stratification(df: pd.DataFrame, cv_folds: int, which_folds: list[int]):
    def _structures_by_id(_df: pd.DataFrame, idlist):
        return _df.query(f"{STRUCTURE_ID} in @idlist")

    if cv_folds < 1:
        raise ValueError(f"cv_folds must be at least 1. {cv_folds} provided")
    df[STRUCTURE_ID] = [get_structure_id_from_filepath(x) for x in df["npy_path"]]
    structure_ids = np.array(df[STRUCTURE_ID].unique().tolist())
    if cv_folds == 1:
        trainval_structure_ids, test_structure_ids = train_test_split(structure_ids, test_size=0.2)
        train_structure_ids, val_structure_ids = train_test_split(trainval_structure_ids, test_size=0.2)
        df_train = _structures_by_id(_df=df, idlist=train_structure_ids)
        df_val = _structures_by_id(_df=df, idlist=val_structure_ids)
        df_test = _structures_by_id(_df=df, idlist=test_structure_ids)
        yield df_train, df_val, df_test, 0
    else:
        kf = KFold(n_splits=cv_folds)
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(structure_ids)):
            if fold_id in which_folds:
                # here trainval_idx & test_idx are indices of structure_ids array!
                train_idx, val_idx = train_test_split(trainval_idx, test_size=(1 - 0.2) * 0.2)
                train_structure_ids = structure_ids[train_idx]
                val_structure_ids = structure_ids[val_idx]
                test_structure_ids = structure_ids[test_idx]
                assert (
                    len(set(train_structure_ids).intersection(val_structure_ids)) == 0
                ), "Overlap between train and val"
                assert (
                    len(set(train_structure_ids).intersection(test_structure_ids)) == 0
                ), "Overlap between train and test"
                assert len(set(val_structure_ids).intersection(test_structure_ids)) == 0, "Overlap between val and test"
                df_train = _structures_by_id(_df=df, idlist=train_structure_ids).drop([STRUCTURE_ID], axis=1)
                df_val = _structures_by_id(_df=df, idlist=val_structure_ids).drop([STRUCTURE_ID], axis=1)
                df_test = _structures_by_id(_df=df, idlist=test_structure_ids).drop([STRUCTURE_ID], axis=1)

                yield df_train, df_val, df_test, fold_id
