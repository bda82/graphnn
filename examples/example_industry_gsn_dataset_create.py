import logging

# from gns.dataset.tech_dataset import tech_dataset_fabric
from gns.dataset.arango_tech_dataset import arango_tech_dataset_fabric

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Init dataset object

logger.info("Init dataset object...")

# ds = tech_dataset_fabric()
ds = arango_tech_dataset_fabric()

# Download dataset

logger.info("Download dataset...")

ds.download()

# Read content of dataset

logger.info("Read content of dataset...")

out = ds.read()

# Print result
print(out)
