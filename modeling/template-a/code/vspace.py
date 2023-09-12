import logging

import numpy as np

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True)
