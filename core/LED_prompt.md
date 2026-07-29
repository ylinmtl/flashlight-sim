Act as an expert optical physicist and data scientist. I am building a hardware database for a custom Finite Element Analysis (FEA) flashlight ray-tracing simulator. I need you to extract physical specifications from the provided LED datasheet (or test plots) and calculate specific mathematical coefficients for its electrical and luminous performance curves.

The Physics Engine Models
My simulator uses the following exact formulas to model the diode physics.
Forward Voltage Curve:
$$V(I) = V_{turn\_on} + V_{scale} \cdot \ln(I + 1.0)$$

Luminous Efficacy Curve (lm/W):
$$E(I) = \text{base\_efficacy\_lm\_w} \cdot \exp(-\text{droop\_factor} \cdot I)$$

Total Lumen Output:
$$\text{Lumens}(I) = I \cdot V(I) \cdot E(I)

$$Your Task Instructions
Step 1: Data ExtractionExtract the raw data points for Current (A) vs. Voltage (V) and Current (A) vs. Luminous Flux (lm) from the provided files or images.
Step 2: Curve FittingCalculate Power ($W = I \cdot V$) and Efficacy ($\text{lm}/W$) for your extracted data points. Then, mathematically fit the data to my specific equations above to solve for these four exact coefficients:vf_turn_on_vvf_scalebase_efficacy_lm_wdroop_factor
Step 3: Physical ParametersDetermine the physical constraints of the LED based on the datasheet. If it is a domeless (HI) emitter, set dome_size_mm to 0.0 and refractive_index to 1.0.

Step 4: JSON OutputOutput the final compiled data strictly in the following JSON format. Do not alter the keys.JSON"EMITTER_NAME_HERE": {
    "max_current_amps": 0.0,
    "vf_turn_on_v": 0.000,
    "vf_scale": 0.000,
    "base_efficacy_lm_w": 0.0,
    "droop_factor": 0.000,
    "footprint_x_mm": 0.0,
    "footprint_y_mm": 0.0,
    "height_mm": 0.0,
    "shape": "square",
    "dome_size_mm": 0.0,
    "refractive_index": 1.0
    "die_length_mm": 0.0,
    "die_width_mm": 0.0,
}