from training import estimate_threshold
from testing import test_model

training_folder = "training_data/"
test_folder = "test_data/"
slice_index = 10

# Train: estimate threshold
pipelines = ['pipeline_1', 'pipeline_2', 'pipeline_3'] #may be slow!
evaluation_results = []
for pipeline in pipelines:
    print(f"Running {pipeline}")
    threshold = estimate_threshold(training_folder, slice_index=slice_index, pipeline=pipeline)
    results = test_model(test_folder, threshold, slice_index=slice_index, pipeline=pipeline)
    if results:
        avg_iou = sum(item['IoU'] for item in results) / len(results)
    else:
        print(f"Warning: No results found for folder {test_folder} with pipeline {pipeline}.")
        avg_iou = 0.0
    evaluation_results.append({"pipeline": pipeline, "threshold": threshold, "average_iou": avg_iou})

print("\nSummary of Evaluation Results:")
for res in evaluation_results:
    print(f"Pipeline: {res['pipeline']}, Threshold: {res['threshold']}, Average IoU: {res['average_iou']:.4f}")