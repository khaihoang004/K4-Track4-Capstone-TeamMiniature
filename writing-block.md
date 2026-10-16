# Task: Implement LaneGuard AI — Pretrained Lane Detection & Lane Departure Warning

You are working inside my existing personal repository. Your task is to implement a working **LaneGuard AI prototype for real-time lane detection and lane departure warning using a live webcam**.

## 1. First inspect the existing repository

Before modifying anything:

1. Inspect the repository structure, README, dependencies, entry points, and existing code.
2. Identify the current implementation of the webcam pipeline and lane detection, if any.
3. Check the existing Python version, virtual environment instructions, and coding conventions.
4. Reuse existing modules where practical. Do not create a separate project or duplicate functionality unnecessarily.
5. Provide a short implementation plan based on the actual repository.

Do not assume file names, directory structures, or dependencies before inspecting the repository.

## 2. Project objective

Build a real-time ADAS-inspired prototype that:

- Reads frames from a webcam.
- Uses a **pretrained lane detection model**, without training from scratch.
- Detects the left and right lane boundaries when possible.
- Estimates the vehicle/camera's relative position within the lane.
- Detects potential lane departure.
- Displays a visual warning in real time.
- Handles unreliable lane detection without producing misleading warnings.
- Reports runtime performance and supports a reproducible demo.

This is a computer vision prototype, not a production automotive safety system. Do not claim that it is safe for real driving.

## 3. Model selection

Preferred initial implementation:

**Ultra-Fast Lane Detection with ONNX Runtime**

Reference repository:
https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference

Official upstream repository:
https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

Requirements:

1. Inspect the reference repository and its README before implementing the integration.
2. Prefer a compatible pretrained ONNX model and its official inference pipeline.
3. Reuse the model's official preprocessing and postprocessing logic whenever possible.
4. Do not train a model from scratch.
5. Do not invent model download URLs, checkpoint names, input dimensions, normalization values, or output tensor formats. Verify these from the upstream source.
6. If pretrained weights cannot be downloaded automatically, provide clear instructions for obtaining and placing them locally.
7. Keep model loading separate from the lane departure warning logic.
8. Do not commit large model weights to Git unless the existing repository explicitly requires this.
9. If the existing repository already contains a working lane detection model, evaluate whether it should be reused before introducing another model.

The application must clearly report missing weights, incompatible models, dependency problems, and webcam initialization errors.

## 4. Architecture

Implement a modular pipeline consistent with the existing repository.

Suggested logical components:

- `CameraSource`: reads webcam frames.
- `LaneDetector`: loads the pretrained model and returns detected lane points or boundaries.
- `LaneGeometry`: estimates the left/right boundaries and lane center.
- `DepartureMonitor`: computes lane offset and determines the warning state.
- `TemporalSmoother`: stabilizes predictions over time.
- `OverlayRenderer`: draws lane boundaries, offset, state, and performance information.
- `Application`: coordinates the pipeline and manages resources.

These are logical responsibilities, not mandatory filenames. Adapt them to the repository's existing architecture.

Avoid overengineering. The goal is a maintainable demo that can be completed and run reliably.

## 5. Lane geometry and offset

Use detected lane points to estimate the lane boundaries at a suitable look-ahead row in the image.

Let:

- `x_left`: estimated left lane boundary.
- `x_right`: estimated right lane boundary.
- `x_vehicle`: image reference point representing the camera/vehicle position.

Calculate:

```python
lane_center = (x_left + x_right) / 2
lane_half_width = (x_right - x_left) / 2

offset = (x_vehicle - lane_center) / lane_half_width
```

Validate that the boundaries are ordered correctly and that the lane width is plausible before calculating the offset.

Interpretation:

- `offset ≈ 0`: the reference point is near the estimated lane center.
- Positive and negative offsets indicate opposite directions according to the chosen coordinate convention.
- Larger absolute offsets indicate a greater relative displacement from the lane center.

Document the sign convention clearly in the code and UI.

This is an image-relative estimate, not a calibrated real-world vehicle position. Do not present it in meters unless a valid calibration method is implemented.

If the lane boundaries are curved, use an appropriate local estimate or fitted curve rather than assuming that every lane is a straight line.

## 6. Warning states

Implement at least three explicit states:

- `NORMAL`: lane geometry is reliable and the estimated offset is below the warning threshold.
- `LANE DEPARTURE WARNING`: the offset exceeds the configured threshold and satisfies the temporal warning condition.
- `LANE NOT RELIABLE`: the detector cannot provide sufficiently reliable lane geometry.

Requirements:

1. Make warning thresholds configurable.
2. Use a configurable number of consecutive frames or a time-based persistence condition to avoid flickering.
3. Smooth the offset across frames, while resetting or degrading the smoothing state appropriately when detection becomes unreliable.
4. Do not interpret a missing lane boundary as a normal driving state.
5. Do not issue a departure warning solely because a single frame has a noisy prediction.
6. Display the current state prominently.
7. Document that this prototype does not estimate vehicle dynamics or provide safety-certified warnings.

Do not describe the system as a complete ADAS or autonomous driving system.

## 7. Live webcam demo

The application should:

- Open a configurable webcam device, defaulting to camera index `0`.
- Display the original camera frame with lane overlays.
- Draw the detected left/right lane boundaries and estimated lane center.
- Display the warning state, normalized offset, FPS, and inference latency.
- Recover gracefully from temporary detection failures.
- Release the camera and other resources when the user exits.
- Support an appropriate keyboard shortcut to quit.
- Avoid accumulating frames in a queue that causes the displayed output to lag significantly behind the live camera.

If the existing repository has a different UI framework, follow its conventions instead of introducing an unnecessary new one.

For the demo, the webcam may point at a monitor playing prerecorded dashcam footage. Clearly distinguish this simulated setup from testing on a real moving vehicle.

## 8. Runtime performance

Measure at least:

- End-to-end processing latency per frame.
- Inference latency, if practical.
- Average FPS over a rolling time window.

Use `time.perf_counter()` or an equivalent monotonic high-resolution timer.

Do not report hardcoded FPS or benchmark results. Measure actual runtime values.

Keep inference execution configurable where feasible. CPU inference should remain a fallback if GPU execution is unavailable, provided the selected ONNX Runtime installation supports it.

Do not assume that installing a GPU-enabled package automatically guarantees GPU execution. Report the configured execution provider and handle unavailable providers clearly.

## 9. Evaluation and evidence

Provide a simple, reproducible evaluation procedure.

The prototype should be testable using both live webcam input and prerecorded road videos, if the existing architecture supports video input without major changes.

Record or document:

- Number of test clips or scenarios.
- Number of manually labeled warning and normal samples.
- True positives, false positives, true negatives, and false negatives.
- Warning precision and recall when the labels support these calculations.
- FPS and latency.
- At least one observed failure case.

Do not fabricate evaluation results. If no evaluation data has been run, provide the evaluation script or procedure and mark the results as pending.

Suggested test scenarios:

1. Clearly visible lane markings.
2. Curved roads.
3. Faded or occluded lane markings.
4. Shadows or road markings that may be mistaken for lane boundaries.
5. Temporary loss of one or both lane boundaries.

Keep the lane detection benchmark metrics separate from the departure-warning metrics.

## 10. Dependencies and model files

Update the existing dependency files appropriately.

Requirements:

- Avoid unnecessary dependency changes.
- Pin versions only when justified by compatibility.
- Keep model weights outside source code.
- Provide a clear setup procedure for the repository's supported operating system.
- Explain how to obtain pretrained weights and where to place them.
- Add appropriate entries to `.gitignore` for virtual environments, caches, logs, and large generated files.
- Do not overwrite existing configuration or dependency files without reviewing their contents.

Prefer a simple installation and launch workflow, such as a documented requirements installation followed by one application entry point.

## 11. Documentation

Update the existing README or add a focused project document covering:

1. Project objective and scope.
2. Architecture and data flow.
3. Pretrained model source and license.
4. Installation and setup.
5. Pretrained weight download and placement.
6. Webcam launch instructions.
7. Configuration options.
8. Warning-state definitions.
9. How to test with prerecorded road videos.
10. Evaluation metrics and how to collect evidence.
11. Known limitations and failure cases.
12. Safety disclaimer.

Explain the Track 4 pillars demonstrated by the implementation:

- **Geometry:** lane boundaries, lane center, and normalized offset.
- **Time:** temporal smoothing and persistent warnings.
- **Uncertainty:** explicit unreliable-detection state.
- **Evidence:** measured latency/FPS, test scenarios, and recorded failure cases.

Only claim a pillar is implemented if the corresponding behavior actually exists in the code.

## 12. Testing and implementation workflow

Work incrementally:

1. Inspect the repository and report the plan.
2. Implement pretrained inference integration.
3. Verify inference on a single image or prerecorded frame.
4. Implement lane geometry and warning-state logic.
5. Integrate the live webcam pipeline.
6. Add runtime metrics and failure handling.
7. Add or update tests and documentation.
8. Run the available tests and smoke checks.
9. Fix regressions introduced by your changes.

At minimum, test:

- Correct offset calculation.
- Invalid or reversed lane boundaries.
- Missing lane boundaries.
- Transition into and out of the warning state.
- Temporal persistence and smoothing.
- Model-loading failure.
- Webcam initialization failure, where testable.

Use mocked detections for unit tests when a real model or camera is not available.

Do not claim tests passed unless they were actually executed.

## 13. Constraints

- Work inside the current repository.
- Preserve unrelated existing functionality.
- Do not delete or rewrite unrelated modules.
- Do not train a model from scratch.
- Do not add unrelated features such as OCR, object detection, driver identity recognition, or vehicle control.
- Do not require a real car for the demo.
- Do not fabricate benchmark results.
- Do not claim safety certification.
- Do not commit secrets, private recordings, or large model weights.
- Keep changes focused and explain important design decisions.

## 14. Final report

After implementation, report:

1. What you discovered in the repository.
2. Which pretrained model and weights were used.
3. Files created or modified.
4. Exact setup and launch commands.
5. Whether live webcam inference was actually tested.
6. Tests executed and their real outcomes.
7. Measured FPS and latency, if available.
8. Known limitations and remaining work.
9. Any manual steps I must complete.

If an external download, camera, or GPU is unavailable, do not pretend the implementation was fully verified. Finish all work that can be completed locally and provide exact next steps.

Begin by inspecting the existing repository. Then implement the smallest reliable end-to-end version of LaneGuard AI.