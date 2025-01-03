from training import estimate_threshold
from testing import test_model

training_folder = "training_data/"
test_folder = "test_data/"
slice_index = 10

# Train: estimate threshold
pipeline = 'pipeline_2'  # Choose the pipeline
threshold = estimate_threshold(training_folder, slice_index=slice_index, pipeline=pipeline)
print(f"Estimated Threshold: {threshold}")

# Test: calculate IoU
results = test_model(test_folder, threshold, slice_index=slice_index, pipeline=pipeline)
for file, iou in results:
    print(f"File: {file}, IoU: {iou:.4f}")