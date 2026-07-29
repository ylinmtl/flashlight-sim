Act as an expert optical engineer and data scientist. I need you to extract physical specifications from the provided LED datasheet (or test plots) and calculate specific mathematical coefficients for its electrical and luminous performance curves.

The Physics Engine Models

My simulator uses the following exact formulas to model the diode physics.

Forward Voltage Curve:

$$
V(I)=V_0+V_s\\cdot ln(I+1.0)
$$

where

* $V_0 =$ `vf_turn_on_v`
* $V_s =$ `vf_scale`

Luminous Efficacy Curve (lm/W):

$$
E(I)=E_0\cdot e^{-kI}
$$

where

* $E_0 =$ `base_efficacy_lm_w`
* $k =$ `droop_factor`

Total Lumen Output:

$$
\mathrm{Lumens}(I)=I\cdot V(I)\cdot E(I)
$$

Your Task Instructions

Step 1: Data Extraction

Extract the raw data points for Current (A) vs. Voltage (V) and Current (A) vs. Luminous Flux (lm) from the provided files or images.

Step 2: Curve Fitting

Calculate Power ($W=I\cdot V$) and Efficacy ($\mathrm{lm}/W$) for your extracted data points. Then, mathematically fit the data to my specific equations above to solve for these four exact coefficients:

* `vf_turn_on_v`
* `vf_scale`
* `base_efficacy_lm_w`
* `droop_factor`
