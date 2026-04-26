# FaceMark

FaceMark is a desktop app for building a face library from local images,
and then using that library to scan a folder for photos containing those people.


UI - `wxPython`<br>
Face detection - `insightface`
## Supported Formats
`.bmp`, `.png`, `.jpeg`, `.jpg`

## Requirements
`requirements.txt`<br>
For CUDA hardware acceleration, follow the documentation `onnxruntime-gpu`.<br>
Without a compatible NVIDIA GPU and cuDNN installation, it should fall back to
CPU inference, if that doesn't work, use `onnxruntime` instead.
