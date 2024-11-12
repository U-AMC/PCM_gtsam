# PCM_gtsam
Visual representation and example of pairwise consistency maximization of factor-graph (Pose-graph). Utilizes Networkx, GTSAM, python for optimization.
PCM requires several assumption and metric to determine 
if two inter-robot measurement ![z_ik^ab](https://quicklatex.com/cache3/18/ql_0f7ae946f34e0b67e1063957885bff18_l3.png) ![z_jl^ab](https://quicklatex.com/cache3/28/ql_2d9ea95b31a0488b22812bcd2c8e5f28_l3.png) are pairwise consistent
Covariance function of inter-robot loop-pair is defined as following:
<p align="center">
$\mathcal{C} \left( \mathbf{z}_{ik}^{ab}, \mathbf{z}_{jl}^{ab} \right) = \left\lVert \left( \ominus \mathbf{z}_{ik}^{ab} \right) \oplus \hat{\mathbf{x}}_{ij}^{a} \oplus \mathbf{z}_{jl}^{ab} \oplus \hat{\mathbf{x}}_{lk}^{b} \right\rVert_{\Sigma} \triangleq \left\lVert \boldsymbol{\epsilon}_{ikjl} \right\rVert_{\Sigma_{ikjl}}$
</p>


### GTSAM Version of Covariance Propagation
<p align="center">
  <img src="figure/gt_propagation.png" />
</p>



### Networkx Version of Covariance Propagation
<p align="center">
  <img src="figure/propagation.png" />
</p>


### TODO
Update version with symforce

Cross-check if this concept is right 


---
For those who seek reference


Studying PGO concept with [nano-pgo](https://github.com/gisbi-kim/nano-pgo) from [Giseop Kim](https://github.com/gisbi-kim)
```
R. Smith, M. Self and P. Cheeseman,
"Estimating uncertain spatial relationships in robotics," Proceedings.
 1987 IEEE International Conference on Robotics and Automation, Raleigh, NC, USA, 1987, pp. 850-850,
 doi: 10.1109/ROBOT.1987.1087846. keywords: {Robots},
```
