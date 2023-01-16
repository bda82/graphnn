import logging

from gns.dataset.arango_dataset import ArangoDataset

logger = logging.getLogger(__name__)

# Init dataset object

logger.info("Init dataset object...")

ds = ArangoDataset(a=1)

# Download dataset

logger.info("Download dataset...")

ds.download()

# Read content of dataset

logger.info("Read content of dataset...")

out = ds.read()

# Print result
print(out)
