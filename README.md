# Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data

Corresponding code for the paper: "Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data", at Network and Distributed System Security (NDSS) 2024.

A guide to the code is available [here](how_to.md).

## Examples

### Static Triggers

![static](./figures/static/static_1.gif)

![static](./figures/static/static_2.gif)

![static](./figures/static/static_3.gif)

![static](./figures/static/static_4.gif)

### Moving Triggers

![moving](./figures/moving/moving_1.gif)

![moving](./figures/moving/moving_2.gif)

![moving](./figures/moving/moving_3.gif)

### Smart Triggers

#### Clean Image

![clean image](./figures/smart/clean_smart.gif)

#### Trigger in the least important area

![trigger](./figures/smart/least_smart.gif)

#### Trigger in the most important area

![trigger](./figures/smart/most_smart.gif)

### Dynamic Triggers

#### Attack Overview

![attack](./figures/dynamic/attack_overview.gif)

#### Dynamic Examples

|       γ            |   0.1	|   0.05	|  0.01 	|
|------------------	|---	|---	|---	|
| Clean image 	|  ![clean image](./figures/dynamic/clean_0.1_dynamic.gif) | ![clean image](./figures/dynamic/clean_0.05_dynamic.gif) 	|   ![clean image](./figures/dynamic/clean_0.01_dynamic.gif)	|
| Noise            	|   ![noise](./figures/dynamic/noise_0.1_dynamic.gif)	|  ![noise](./figures/dynamic/noise_0.05_dynamic.gif) 	|  ![noise](./figures/dynamic/noise_0.01_dynamic.gif) 	|
| Projected  Noise 	|  ![projection](./figures/dynamic/noise_proj_0.1_dynamic.gif) 	|   ![projection](./figures/dynamic/noise_proj_0.05_dynamic.gif)	|  ![projection](./figures/dynamic/noise_proj_0.01_dynamic.gif) 	|
| Backdoor image   	|   ![bk image](./figures/dynamic/bk_0.1_dynamic.gif)	|   ![bk image](./figures/dynamic/bk_0.05_dynamic.gif)	|   ![bk image](./figures/dynamic/bk_0.01_dynamic.gif)	|


## Authors

Gorka Abad, Oguzhan Ersoy, Stjepan Picek, and Aitor Urbieta.

## How to cite

```
@inproceedings{abad2024sneaky,
  title={Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data.},
  author={Abad, Gorka and Ersoy, Oguzhan and Picek, Stjepan and Urbieta, Aitor.},
  booktitle={NDSS},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
