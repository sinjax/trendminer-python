from bivariate.tools.utils import *
from ..dataset.userwordgender import *
from IPython import embed
import logging;logger = logging.getLogger("root")

region_root = "/data/ss/UK_regions_Sina/prepared/X_expand_scipy"
gender_root = "/data/ss/UK_regions_Sina/prepared/X_expand_scipy_gender"
gender_map_file = "/data/ss/UK_regions_Sina/gender.users"

gender_map = load_gender_map(gender_map_file)

region_to_gender(region_root,gender_root,gender_map)