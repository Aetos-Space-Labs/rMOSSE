Fast MOSSE multi-tracker implementation aimed for resource-constrained edge devices.  

High performance is mainly achieved by using SIMD, data parallelism and caching.  

Besides basic single-object tracking we have an efficient multi-tracking interface.

### Synthetic init/update timing  (lower is better)

| Implementation | Init&nbsp;(ms) | Update&nbsp;(ms) |
|---------------:|--------------:|-----------------:|
| **rMOSSE** | **31.93** | **4.33** |
| OpenCV baseline | 61.00 | 12.63 |

### Visual comparison

| rMOSSE (using a single multi-tracker) | OpenCV baseline (using multiple trackers) |
|:----------:|:---------------:|
| <img src="https://github.com/user-attachments/assets/16101211-4cb2-41a3-a12f-dd4eb77ad3f9" width="320"> | <img src="https://github.com/user-attachments/assets/b5b11d51-4d54-4c8d-8e30-e7b18adb94af" width="320"> |

Refernce paper: https://ieeexplore.ieee.org/document/5539960