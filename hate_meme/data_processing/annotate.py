import io
import json
import os
from datetime import datetime

import ipywidgets as widgets
from IPython.display import display, Image, clear_output
from PIL import Image as PILImage

def display_fixed_image(img_path, size=(300, 300), bg_color=(255, 255, 255)):
    """Display a fixed-size image in a Jupyter notebook with a white background."""
    with PILImage.open(img_path) as img:
        img = img.convert("RGB")
        img.thumbnail(size, PILImage.LANCZOS)
        background = PILImage.new("RGB", size, bg_color)
        offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
        background.paste(img, offset)
        buf = io.BytesIO()
        background.save(buf, format='PNG')
        buf.seek(0)
        display(Image(data=buf.read(), width=size[0], height=size[1]))


def load_existing_labels(output_file, annotator):
    labeled = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('annotator') == annotator:
                        labeled.setdefault(entry['id'], {})
                        if 'label_incivility' in entry:
                            labeled[entry['id']]['incivility'] = True
                        if 'label_intolerance' in entry:
                            labeled[entry['id']]['intolerance'] = True
                except Exception:
                    continue
    return labeled


def annotate_data(df, output_file, annotator):
    """
    Annotate images in a Jupyter notebook for two categories: incivility and intolerance.

    Args:
        df: pandas DataFrame with at least columns 'id' and 'img_path'
        output_file: path to output JSON file
        annotator: string, annotator's name
    """
    idx = 0
    out = widgets.Output()

    labeled = load_existing_labels(output_file, annotator)

    def needs_annotation(row):
        entry = labeled.get(int(row['id']), {})
        return not (entry.get('incivility') and entry.get('intolerance'))

    to_annotate = df[df.apply(needs_annotation, axis=1)].reset_index(drop=True)
    n_to_annotate = len(to_annotate)

    def save_annotation(row, label_incivility, label_intolerance):
        entry = {
            "id": int(row['id']),
            "annotator": str(annotator),
            "label_incivility": int(label_incivility),
            "label_intolerance": int(label_intolerance),
            "label_hateful": int(row['label']),
            "text": row['text'],
            "time": datetime.now().isoformat(),
            "img": str(row['img']),
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def show_next(change=None):
        nonlocal idx
        out.clear_output()
        if idx >= n_to_annotate:
            with out:
                print("All images labeled!")
            return

        row = to_annotate.iloc[idx]
        img_path = row['img_path']

        text_box_incivility = widgets.Text(description="Incivility:", placeholder="Enter label (number)")
        text_box_intolerance = widgets.Text(description="Intolerance:", placeholder="Enter label (number)")
        btn_next = widgets.Button(description="Next", button_style='success')

        # State to track if both categories are labeled
        state = {'incivility': None, 'intolerance': None}

        def update_next_button(*args):
            btn_next.disabled = not (
                state['incivility'] is not None and state['intolerance'] is not None
            )

        def on_incivility_change(change):
            try:
                state['incivility'] = int(change['new'])
            except ValueError:
                state['incivility'] = None
            update_next_button()

        def on_intolerance_change(change):
            try:
                state['intolerance'] = int(change['new'])
            except ValueError:
                state['intolerance'] = None
            update_next_button()

        def on_next_clicked(b):
            save_annotation(row, state['incivility'], state['intolerance'])
            next_image()

        def next_image():
            nonlocal idx
            idx += 1
            show_next()

        text_box_incivility.observe(on_incivility_change, names='value')
        text_box_intolerance.observe(on_intolerance_change, names='value')
        btn_next.on_click(on_next_clicked)
        btn_next.disabled = True  # Initially disabled

        with out:
            clear_output(wait=True)
            print(f"Annotator: {annotator}")
            print(f"Image {idx+1}/{n_to_annotate} (ID: {row['id']})")
            try:
                display_fixed_image(img_path, size=(512, 512))
            except FileNotFoundError:
                next_image()
                return
            display(widgets.HBox([text_box_incivility, text_box_intolerance, btn_next]))

    show_next()
    display(out)