![Logo](<repo/Frame 6.png>)
# Vektor Engine (early beta)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
Open Source game engine inspired by UE rendering techniques like ReSTIR DI from MegaLights, Hybrid RTGI from Lumen. Engine is fully royality free, and anyone can edit it without even credits to dev.



## Features

- Realtime Ray Tracing with 1000+ lights via Solar DI (Optimized ReSTIR), RTGI
- Easy scripting language .SCR
- Optimized Editor
- Vulkan Crossplatform
- NRD Denoiser

## Demo

In shown example we loaded de_dust2 voxel scene, due to Solar DI and optimized RTGI we can place magnitudes of lights. For example in our scene we placed 1000 lights (3 additional lights is directional, sky, and area emission shape in sky) we getting extremely good results in our case, remember all lights are fully dynamic with ray traced shadows and dynamic bounces.

![Demo](<repo/Знімок екрана 2026-11-16 181037.png>)


## Technical Breakdown

Engine used special technologies to get realestic lighting, and good perfomance. In this image you can find a simple breakdown:

![Logo](<repo/Frame 7.png>)

## Building

Requirements:

GPU driver supporting VK_KHR_ray_tracing_pipeline and VK_KHR_acceleration_structure
(optional) Nvidia RTX 4080 or 5060 for DLSS 4.5 support

```bash
cmake --build build --config Release
```
Then to run engine main module type:

```bash
build\Release\rt_renderer.exe --sun
```
CMAKE will automatically download all dependencies from their repositories

## Scripting
Documentation soon!

