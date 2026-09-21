# EHT Analytica

Deep learning EHT (Engineered Heart Tissue) pillar tracking and contraction
analysis, distributed to wet-lab users as a desktop application.

## Language

### Distribution

**Installer**:
The single small executable a user downloads from GitHub; it downloads the
Payload and assembles the Installation.
_Avoid_: setup, downloader, updater

**Payload**:
The set of split archive files published as GitHub release assets that
together contain the complete application, including the GPU inference
runtime.
_Avoid_: package, bundle

**Volume**:
One of the split archive files that together make up the Payload.
_Avoid_: part, chunk, segment

**Installation**:
The assembled, runnable application directory on the user's machine, produced
by the Installer from the Payload.
_Avoid_: install folder, app dir

### Analysis workflow

**Inference Sample**:
An ROI image together with its model-predicted pillar coordinates, saved
during a tracking run (when "Save inferences" is enabled) for later human
review as training data.
_Avoid_: training sample, inference artifact
