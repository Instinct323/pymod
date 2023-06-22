import logging
import pickle
from pathlib import Path

ROOT = Path('../tmp').resolve()
INFO = ROOT.parent / 'info'
SOURSE = ROOT / 'raw_data.pkf'

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.info(f'cwd: {INFO}\n')


def dataframe_set(df):
    ser = {key: set(df[key]) for key in df}
    return pd.Series(ser)


def lazy_obj(file, fget, *args, **kwargs):
    if file.is_file():
        with open(file, 'rb') as pkf:
            data = pickle.load(pkf)
    else:
        data = fget(*args, **kwargs)
        with open(file, 'wb') as pkf:
            pickle.dump(data, pkf)
    return data


_build_fcn = lambda: NotImplemented
DATA = lazy_obj(SOURSE, _build_fcn)