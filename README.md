# RetinaNet-project

Keras implenmentation of RetinaNet. \
![alt text](https://github.com/kwdaisuke/RetinaNet-project/blob/main/data/grand.png?raw=true)

with prediction scores \
![](https://github.com/kwdaisuke/RetinaNet-project/blob/main/data/Screenshot%20(102).png?raw=true)

# Update
- Add ImageLoader method 3/13
- Add Argumentparser 3/14

Make an inference of images which are in a folder typed as argument of command line
![](data/argparser.png)

Terminal shows the result of inference with float number \
![](data/terminal.png)


# Todo
- Try different benchmarks
- Try with different layers
- Normalization free model → Poor Performance with pre-made module. 

# References:
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. ICCV, 2017.
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144). Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. CVPR, 2017.
- Keras RetinaNet: https://github.com/fizyr/keras-retinanet
