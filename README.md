# WCPSP_Stereo
### Weighted Cost Propagation with Smoothness Prior for fast stereo correspondence

WCPSP_Stereo is a fast stereo correspondence algorithm utilizing *weighted cost
propagation with smoothness prior* on horizontal tree strucutres introduced in
our work "Local Smoothness Enforced Cost Volume Regularization for Fast Stereo
Correspondence".

Note: After the publication of the paper, we changed to the KITTI datasets.  The
matching cost computing function is different as that is described in the paper.
The code for cost computation is adapt from the code for the paper "Efficient
Joint Segmentation, Occlusion Labeling" by Koichiro Yamaguchi et
al.(http://ttic.uchicago.edu/~dmcallester/SPS/index.html).

### Building
1. Prerequisites
    * [libpng](http://www.libpng.org/pub/png/libpng.html)
    * [png++](http://www.nongnu.org/pngpp/)
2. Building
    1. create a folder 'mkdir build' and 'cd build'
    2. type 'cmake ..'
    3. type 'make'

### Usage of the demo code

### License
WCPSP-Stereo is licensed under the The MIT "Expat" License.