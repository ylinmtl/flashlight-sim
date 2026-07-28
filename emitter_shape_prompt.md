# Emitter input generator prompt

Paste everything between the two rules into an AI assistant, together with a
datasheet PDF, a photo of the emitter, or just a written description. It
returns a catalogue entry ready to paste into `hardware_library.json`.

Check the result with `emitter_shapes.py` before trusting it. An assistant
reading a photo will guess at anything the picture does not show, and a die
outline that is plausible but wrong still runs.

---

You are helping build an input file for an optical ray tracing simulator that
models flashlight beams. I will give you a datasheet, a photograph, or a
description of an LED emitter. Produce a single JSON object describing it.

## Coordinate conventions

These are not negotiable; the simulator depends on them.

- All lengths are in **millimetres**.
- The **die outline** is drawn in a plane looking down at the emitter from
  above, with **+x to the right** and **+y up**.
- The **origin is the centre of the light emitting surface**, which the
  simulator places on the reflector's optical axis. The outline must be
  centred on it: the bounding box centre must be (0, 0) to within about 1% of
  the die size.
- Outline vertices are listed **in order around the perimeter**, either
  clockwise or anticlockwise. Do **not** repeat the first vertex at the end;
  the outline closes automatically.

## The `shape` field

Pick one of exactly three values:

| `shape` | Use it when | Also needs |
|---|---|---|
| `"square"` | The emitting area is a plain rectangle or square | `die_length_mm`, `die_width_mm` |
| `"round"` | The emitting area is a true circle | `die_length_mm` as the diameter |
| `"polygon"` | Anything else at all | `die_outline` |

Use `"polygon"` for chamfered corners, corner notches, rounded corners,
hexagons, crosses, multiple separated segments approximated as one outline, or
any irregular shape. When in doubt, use `"polygon"`: it can represent a
rectangle exactly, so it is never the wrong answer.

Two common cases, both real:

- **Chamfered square**, e.g. Luminus SFT60. A square with each corner sliced
  off at 45 degrees, giving an octagon with 8 vertices.
- **Notched square**, e.g. Luminus SFT40. A square with a small rectangle
  removed from each corner, giving a concave outline with 12 vertices. The
  simulator handles concave outlines correctly, so model the notches properly
  rather than smoothing them away.

## How to get the numbers

**From a datasheet.** Use the mechanical drawing, not the product photo.
Datasheets usually give the package footprint and, separately, the emitting
area or "die size". You want the **emitting area**, which is smaller than the
package. If the drawing dimensions the chamfer or notch, use it directly. If it
does not, measure it off the drawing in proportion to a dimension that *is*
given.

**From a photograph.** Find one length you know, normally the package width,
and scale everything to it. Say in your notes which dimension you used as the
ruler. Photographs are taken at an angle often enough that you should say so if
the die looks skewed.

**From a description.** Ask for whatever is missing rather than inventing it.
It is better to return fewer fields than to fill them with plausible numbers.

## Output format

Return **only** a JSON object, no commentary before or after, in this form:

```json
{
  "Manufacturer Model CCT": {
    "die_length_mm": 2.0,
    "die_width_mm": 2.0,
    "shape": "polygon",
    "die_outline": [[-0.7,-1.0],[0.7,-1.0],[0.7,-0.7],[1.0,-0.7],[1.0,0.7],[0.7,0.7],[0.7,1.0],[-0.7,1.0],[-0.7,0.7],[-1.0,0.7],[-1.0,-0.7],[-0.7,-0.7]],
    "footprint_x_mm": 3.5,
    "footprint_y_mm": 3.5,
    "height_mm": 0.72,
    "max_current_amps": 2.4,
    "vf_turn_on_v": 2.569,
    "vf_scale": 0.622,
    "base_efficacy_lm_w": 213.3,
    "droop_factor": 0.319,
    "dome_size_mm": 3.1,
    "refractive_index": 1.41
  }
}
```

Field meanings:

- `die_length_mm`, `die_width_mm` — bounding box of the emitting area. For a
  polygon these are descriptive; the outline is what gets sampled. Set them to
  the outline's bounding box so they agree.
- `footprint_x_mm`, `footprint_y_mm` — the **package** outline, used for gasket
  clearance. Larger than the die.
- `height_mm` — height of the package above the board, i.e. how far the
  emitting surface sits above the reflector shelf.
- `max_current_amps` — maximum forward current.
- `vf_turn_on_v`, `vf_scale` — forward voltage model, `Vf = vf_turn_on_v +
  vf_scale * sqrt(current)`. Fit these to two points on the datasheet's Vf
  curve if it has one.
- `base_efficacy_lm_w` — lumens per watt near the low end of the current range.
- `droop_factor` — fractional efficacy loss at maximum current relative to the
  base, between 0 and 1. Read it off the relative flux curve.
- `dome_size_mm` — diameter of the silicone dome. Use `0` for a flat or domeless
  emitter such as an HI or SMD part, and `-1` to mean "as wide as the package".
- `refractive_index` — of the dome material, typically 1.41 to 1.55 for
  silicone. Use `1.0` when `dome_size_mm` is `0`.

After the JSON, add a short block listing:

1. Which numbers came straight from the source.
2. Which you estimated, and from what.
3. Anything the source did not show at all.

## Before you answer, check

- Vertices are in order around the perimeter, not scrambled.
- The first vertex is not repeated at the end.
- The bounding box is centred on (0, 0).
- The outline is a few millimetres across, not a few hundred. Microns are a
  common datasheet unit; convert them.
- A concave shape actually goes inward, rather than being simplified away.
- `die_length_mm` and `die_width_mm` match the outline's bounding box.

---

## Verifying the result

```bash
python emitter_shapes.py --list          # what the generator can build directly
python emitter_shapes.py sft60 --size 2.25 --corner 0.34 --preview
```

To check an outline an assistant produced, paste it into a Python session:

```python
import emitter_shapes as es

outline = [[-0.7, -1.0], [0.7, -1.0]]  # ... the full list
print(es.describe(outline))
for problem in es.validate(outline):
    print("[!]", problem)
es.preview(outline, elements_per_side=16)   # draws it with the sample points
```

`validate` catches the mistakes that still produce a runnable simulation:
wrong units, an outline that is not centred, duplicated vertices, and vertices
listed out of order. `preview` draws the outline together with the points the
tracer will actually sample, which is the fastest way to see that a notch or
chamfer landed where you meant it to.
