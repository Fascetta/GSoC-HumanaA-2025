# 🩰 Motion Dance Lab – 3D Mocap Visualization & Labeling Tool

A modular Python project for visualizing and labeling dance motion capture (mocap) sequences. Built with [Open3D](http://www.open3d.org/) and Jupyter widgets, it lets you explore complex motion data in 3D and annotate clips with movement, emotion, and style labels. 

> ⚠️ Note: This project is a **work in progress**, initially built under tight deadlines. Future enhancements are welcome!

---

## ✨ Features

- **3D Joint Animation**: View skeletal motion in a dynamic 3D scene.
- **Custom Clustering**: Color-coded body parts (arms, legs, torso, etc.) for better interpretability.
- **Windowed Chunking**: Prepares overlapping motion chunks for more granular labeling and analysis.
- **Interactive Labeling Tool**: Jupyter-based UI to label motion segments across multiple axes (e.g., emotion, dance style, tempo).

---

## 🧱 Project Structure

```
motion-dance-lab/
├── data/
│   ├── raw/                 # Raw .npy motion files (e.g. full dances)
│   ├── processed/           # Chunked sequences + metadata
│   └── labels/              # User-created labels
├── notebooks/
│   └── 03_labeling_tool.ipynb   # Interactive labeling notebook
├── src/
│   ├── data/
│   │   └── chunker.py       # File chunking and metadata generation
│   └── utils/
│       └── visualization.py # Animation, color logic, rotation
└── README.md
```

---

## 📦 Installation

```bash
pip install numpy open3d ipywidgets matplotlib
```

Also make sure Jupyter Notebook extensions for `ipywidgets` are enabled:

```bash
jupyter nbextension enable --py widgetsnbextension
```

---

## 📁 Usage

### 1. Prepare Raw Data

Place `.npy` mocap files in `data/raw/`. Each file should have shape `(T, J, 3)` where:
- `T`: number of frames
- `J`: number of joints
- `3`: x/y/z coordinates

---

### 2. Chunk Motion Files

Use the `chunker.py` script to create overlapping chunks:

```bash
python src/data/chunker.py
```

This saves:
- Chunked data: `data/processed/yourfile_chunks.npy`
- Metadata: `data/processed/chunk_metadata.json`

---

### 3. Label Dance Segments

Open the labeling tool:

```bash
jupyter notebook notebooks/03_labeling_tool.ipynb
```

- Use slider to scroll through chunks.
- Select multiple labels (movement type, emotion, speed, style).
- Click 💾 Save to store labels in `data/labels/labels.json`.

---

## 🖼️ Visualization Logic

The project uses `Open3D` to render joints as spheres and build a surrounding "ballroom" box for reference. Joint coloring is based on cluster ID (e.g. left leg, torso, head).

To animate a full sequence independently:

```python
from src.utils.visualization import animate_dance, get_joint_color_config
import numpy as np

data = np.load("data/raw/yourfile.npy")
CLUSTERS, _, JOINT_COLORS = get_joint_color_config()
animate_dance(data, CLUSTERS, JOINT_COLORS)
```

---

## 📌 Labels Covered

Categories include:
- **Action**: `walking`, `jumping`, `spinning`, `pose`, etc.
- **Emotion**: `happy`, `sad`, `relaxed`, `dramatic`, etc.
- **Style**: `hiphop`, `ballet`, `freestyle`, `robotic`, etc.
- **Tempo**: `slow`, `medium`, `fast`, `syncopated`
- **Form**: `solo`, `duet`, `floorwork`, `arm focus`, etc.

---

## 🧪 Future Ideas

- Export animations to video/GIF.
- Add gesture detection and auto-labeling.
- Extend labels with multi-annotator support.
- Train classifier on labeled chunks.

---

## 🤝 Contributing

Bug reports, feature requests, or pull requests are welcome. If you'd like to help extend this project, feel free to fork or open an issue.
