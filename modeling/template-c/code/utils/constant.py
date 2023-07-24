from mod.zjdl.utils.utils import Path, LOGGER

ROOT = Path('../tmp').resolve()
INFO = ROOT.parent / 'info'
SOURSE = ROOT / 'raw_data.pkf'

LOGGER.info(f'cwd: {INFO}\n')
