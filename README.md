# **Lightweight 2D Mapping using Monocular Depth (MiDaS-small)**



Lightweight 2D mapping system using monocular depth estimation with **MiDaS-small.**

This repository explores projecting monocular depth maps into a **LiDAR-like 2D representation** and validating **polar coordinate projection** on an occupancy grid.

Depth maps are generated on an **Android smartphone** using MiDaS-small and transmitted to a **CPU-only PC**, where they are projected into a 2D occupancy grid.



## **Project Goal**

This project started with a smartphone and a normal PC without a GPU.

The goal is to create a **low-cost mapping system** using only:

* **a smartphone camera**
* **onboard IMU sensors**
* **CPU-only computation**



The system does **not aim for high accuracy**, but rather to produce a map that is **geometrically recognizable**.



## **Current Status**

This project is unfinished and currently functions as a front-end system only.

* **No loop closure**
* **No global optimization**
* **The point cloud quality is still limited**

At the moment, the generated maps are barely recognizable, so further improvements are required.

You may think of this repository as a **proof-of-concept demo**.



## **Core Idea**

This system is built on the assumption that **low-cost sensors are unreliable**.

Instead of trusting continuous sensor readings, the system only uses **specific motion events**.

The mapping device (smartphone) is constrained to:

* **pitch rotation**
* **translation along the local x-axis**



Translation is detected using **event-like motion pulses,** defined as:

* **short bursts of acceleration**
* **occurring within a short time interval**



The energy of the acceleration pulse is used to estimate the magnitude of the translation.



## **Absolute Scale Calibration**

Before mapping begins, the system must determine an **absolute depth scale**.

The user generates multiple **translation pulses**.
During this process the system looks for synchronization between:

* **acceleration pulses from the IMU**
* **pixel translation spikes detected using ORB**

Once these two signals are synchronized, a scale constant is calculated to convert:

relative depth → absolute depth

After this calibration step, mapping begins.



## **Data Collection Strategy**

Because the system does not fully trust the sensors, it only collects useful data when the device is stationary.

The workflow is therefore:

1. **User performs a short abrupt movement**
2. **Device stops**
3. **System collects stable depth information**
4. **Data is projected into the occupancy grid**

This process must be repeated continuously by the user.



## **Future Work**

Planned improvements include:

* **Improving front-end correction**
* **Generating higher quality point clouds**
* **Implementing loop closure**
* **Upgrading the system from visual odometry → SLAM**
*

## **Current Result**

Before point cloud realignment
Event_driven_MiDas_based_2Dmapping

![VideoProject2-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/0947e797-155c-4c0b-b31c-62f8fbdb5756)


After point cloud realignment
Event_driven_MiDas_based_2Dmappint_v2

![mapping3-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/cd2d6df7-cef0-41be-96f2-89912b459c0e)

Adding realignment helps sparse point cloud to come together and form a shape, but simple realignment without any sense of geometry is not enough to form a decent map.
I am currently working on adding the perception of geometrical structure of the indoor environment to the Depth module, in order for the local point cloud to form a shape of the wall or corner.
By doing this I am expecting the map to form in the shape of a more recognizable indoor environment.

