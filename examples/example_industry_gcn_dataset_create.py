import logging

from gns.dataset.tech_dataset import TechDataset
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer
from gns.transformation.layer_process import layer_process_fabric

logger = logging.getLogger(__name__)

logger.info("Init dataset objects...")

ds1 = TechDataset("jd_data", transforms=[layer_process_fabric(GCNConvolutionalGeneralLayer)]) # 249 nodes
ds2 = TechDataset("jd_data2", transforms=[layer_process_fabric(GCNConvolutionalGeneralLayer)]) # 177 nodes

logger.info("Download datasets...")

ds1.download()
ds2.download()

# Read content of dataset

logger.info("Read content of datasets...")

out1 = ds1.read()
out2 = ds2.read()

# Print result
print(out1)
print(out2)
