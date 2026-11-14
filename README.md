

![Logo](https://i.ibb.co/1t8Q3QSS/fulllogo.png)
# Vektor Engine

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
AAA Open Source game engine inspired by UE rendering techniques like ReSTIR DI, Surface Cache GI, Nanite. Engine is fully royality free.

## Features

- Realtime Ray Tracing with 1000+ lights via ReSTIR DI, RTGI
- Easy scripting language .SCR
- Optimized Editor
- Vulkan Crossplatform
- NRD Denoiser (Nvidia libraries need to be imported manually)
- DLSS 4.5 (Nvidia libraries and NRD need to be imported manually)
- Custom shader support

## Demo

In shown example we loaded de_dust2 voxel scene, due to ReSTIR DI and optimized RTGI we can place magnitudes of lights. We are getting roughly 210 fps. Note that shadows are fully raytraced, with real life like quality.

![Logo](demo.png)

