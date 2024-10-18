import os
import tensorrt as trt

NAME = 'unet'
MODEL_DIR = os.path.expanduser('~/Desktop/Image Shifter/vision_progress')

# Set up the logger (Note that the logger is more verbose in this example)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Initialize the builder, network, and parser
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Load the ONNX model
with open(f'{MODEL_DIR}/{NAME}.onnx', "rb") as model:
    if not parser.parse(model.read()):
        print('Failed to parse the ONNX model.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# Configure the builder
config = builder.create_builder_config()
# config.max_workspace_size = 1 << 30  # 1 GB

# Optional: Enable FP16 if supported
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# Build the engine
engine = builder.build_serialized_network(network, config)

# Save the engine to a file
with open(f'{MODEL_DIR}/{NAME}.trt', "wb") as f:
    f.write(engine)