import logging

# from gns.dataset.arango_sfedu_dataset import arango_sfedu_dataset_fabric
from gns.dataset.tech_dataset import tech_dataset_fabric

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Init dataset object
logger.info("Init dataset object...")
# ds = arango_sfedu_dataset_fabric()
ds = tech_dataset_fabric()

# Download dataset
logger.info("Download dataset...")
ds.download()

# Read content of dataset
logger.info("Read content of dataset...")
out = ds.read()

# Print result
print(out)
