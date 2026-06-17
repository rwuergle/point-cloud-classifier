# Documentation
This is the *point-cloud-classifier* documentation. It describes the main idea of every algorithm for the generation of a classified point cloud. 
Currently, 5 classes are implemented:
* [Ground Class](#ground-class)
* (Large) Vegetation Class
* Roof Class
* Facade Class
* Roof Structures Class
* Vehicles Class

The approach used was mostly based in [geometric features](#geometric-features) without deep learning. Therefore, the computation of geometric features is needed and handled in this package:
* [GeometricFeatureCalculation](#geometricfeaturecalculatorpoint_cloud_path-str-slope-float--none--none)

## 1. Geometric Features
All those geometric attributes first compute the covariance matrix of the subset of points selected know as the 3D structure tensor, whith its eigenvalues $\lambda_1 \geq \lambda_2 \geq \lambda_3\geq 0$ and combine those in a list of geometric indicators as shown below. Those indicators were taken from the paper [Contour Detection in Unstructured 3D Point Clouds](https://openaccess.thecvf.com/content_cvpr_2016/papers/Hackel_Contour_Detection_in_CVPR_2016_paper.pdf) which has as goal to detect contours with a binary classifier, which is not exactly the goal that we persue. Most of the classifier proposed come originally from the paper [Feature Relevance Assessment for The Semantic Interpretation Of 3D Point Cloud Data](https://isprs-annals.copernicus.org/articles/II-5-W2/313/2013/isprsannals-II-5-W2-313-2013.pdf).

### 1.1 Description

For describing the local dimensionality, the measures of linearity $L_\lambda$, planarity, $P_\lambda$ and scatter (i.e. sphericity) $S_\lambda$ provide information about the presence of a linear 1D structure, a planar 2D structure or a volumetric 3D structure:
* $L_\lambda = \frac{\lambda_1 - \lambda_2}{\lambda_1}$
* $P_\lambda = \frac{\lambda_2 - \lambda_3}{\lambda_1}$
* $S_\lambda = \frac{\lambda_3}{\lambda_1}$

Further measures are provided by omnivariance $O_\lambda$, anisotropy $A_\lambda$, eigenentropy $E_\lambda$ and the sum of eigenvalues denoted as $\Sigma_\lambda$. Exploiting the change of curvature denoted as $C_\lambda$ has also been proposed.
 * $O_\lambda = \sqrt[3]{\lambda_1\lambda_2\lambda_3}$
 * $A_\lambda = \frac{\lambda_1-\lambda_3}{\lambda_1}$
 * $E_\lambda = -\sum\limits_{i=1}^3 \lambda_i \ln(\lambda_i)$
 * $\Sigma_\lambda = \lambda_1 + \lambda_2 + \lambda_3$
 * $C_\lambda = \frac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}$

 Other features that can be noted are Verticality V and density D.
 * $V = 1 - n_z$ , where $n_z$ is the third component of the normal vector $n$
 * $D = \frac{k + 1}{\frac{4}{3}\pi r^3_{k_{NN}}}$ , where $r_{k_{NN}}$ represents the radius of the spherical neighborhood defined
by a 3D point and its k closest neighbors.

Some feature are more oriented in the vertical directino such as poles, we can project the 3d points onto a 2d horizontal plane $\mathcal{P}$. Then we can derive the local 2D density based on a $r_{k_{NN}, 2D}$. We then get the 2d eigenvalues and we can also have their ratio $R_{\lambda, 2D}$ as indicator.
* $D_{2D} = \frac{k + 1}{\pi r^2_{k_{NN}, 2D}}$
* $R_{\lambda, 2D} = \frac{\lambda_{2,2D}}{\lambda_{1,2D}}$
* $\Sigma_{\lambda, 2D} = \lambda_{1,2D} + \lambda_{2,2D}$

One can also introduce binning into a raster called accumulation map $\mathcal{M}(X,Y)$ of the 2D plane, that max show the presence of a 3D structure if its bin value is high. From it, we can derive the height difference $\Delta Z$ and the standard deviation $\sigma_z$. One can also imagine that we want to know the relative position of a given point in the vertical structure $\tilde z$.
* $\Delta Z$
* $\sigma_z$
* $\tilde z$

### 1.2 `GeometricFeatureCalculator(point_cloud_path: str, slope: float | None = None)`

The geometric features are computed thanks to the class `GeometricFeatureCalculator`. The constructor needs as input the path to a point cloud with format `(copc).laz` or `(copc).las`. All the possible features are visible in `REQUIRED_FEATURES` for `constants.py`.

#### 1.2.1 `compute_relative_z(feature_names: list[str] = ["z_norm"]) -> None`
This methods computes the relative Z-coordinate (altitude) with respect to a simulated cloth which should find the ground. This is done using the cloth simulation filter described in the paper *[An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation](https://www.mdpi.com/2072-4292/8/6/501)*. The parameters are adjusted based on the mean slope of the tile (works only for canton NE tiles with `\d{7}_\d{7}` format) or the input slope given to the constructor.

```python
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = max(np.round(-0.0226 * self.slope + 1.045), 0.2)
    csf.params.rigidness = 3 if self.slope < 5 else (2 if self.slope < 20 else 1)
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.5
    csf.params.interations = 200 
```
The linear regression on the `cloth_resolution` and `rigidness` conditions are based on empirical tests on NE-tiles. 

#### 1.2.2 `compute_3d_features(features: dict[str, list[float]]) -> None`
This method computes different geometric features (`str`) on a sphere with given radii (`list[float]`). In order to do so, the library [jakteristics](https://github.com/jakarto3d/jakteristics) was used.

#### 1.2.3 `compute_2d_features(features: dict[str, list[float]]) -> None`
This method computes different geometric features (`str`) on a cylinder with $Z \in \left] -\infty, +\infty \right[$ with given radii (`list[float]`). In order to do so, the library [jakteristics](https://github.com/jakarto3d/jakteristics) was used.

#### 1.2.4 `compute_3d_features(features: dict[str, list[float]]) -> None`
This method computes different geometric features (`str`) based on the z-axis distribution of a raster of resolution ($r\times r$) with r (`list[float]`).


## 2. Classes
The classes are all determined one after the other with a binary rule:
* 1: points belong to the class
* 0: points stays unclassified

The data for the classifier is transformed from point cloud to `points` (N x 3D xyz np.ndarray), `data` (all computed features, N x 46 np.ndarray per default) and if there is already a classification, `Y` (N, np.ndarray with labels or 1/0 labelisation). This is done with the static method `DataClassifierFormat.load_data()`

### 2.1 Ground Class
The ground class is separated using a binary random forest classifier on 46 features (geometric features, return number, number of return and intensity). This choice was made to improve the flaws of the cloth simulation filter, but it uses the attribute `z_norm` computed with the cloth simulation filter. The importance of each feature in the pretrained RF classifier can be seen in the plot below:
![random_forest_feature_importance](./plots/ground_RF_feature_importance.png)

