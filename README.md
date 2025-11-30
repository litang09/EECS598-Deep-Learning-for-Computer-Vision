# EECS598

## Version
[EECS598 Winter 2022](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)

## Assignment Details

### A1

- [PyTorch101](Assignment/A1/pytorch101.ipynb): The first assignment of A1 is a coarse introduction to PyTorch which includes the basic utilization of it. More specifically, it will get you familiar with common Pytorch tensor operations like  `torch.tensor` , `torch.arange` , `torch.reshape` , `torch.argmax` ,  `torch.zeros` , `torch.randn` and so on. Aside from that, it also demonstrates how to use **broadcast mechanism** to write vectorized code for GPU to accelerate.

- [kNN](Assignment/A1/kNN.ipynb): In this assignment, you will implement kNN , which is a lazy algorithm that learns no parameter, then you
will use cross-validation to find the best k.

### A2

- [Linear Classifier](Assignment/A2/linear_classifier.ipynb):In this assignment, you will implement a linear classifier and apply **multi-class SVM loss** and **Softmax loss** to it separately. 
  
- [Two-Layer Nets](Assignment/A2/two_layer_net.ipynb): Implement a two-layer neural network. In this assignment, you will explore the role of activation functions in neural networks. Additionally, you will train the model on the CIFAR-10 dataset and tune hyperparameters such as learning rate and regularization strength to achieve optimal accuracy.
  
### A3

- [Fully-Connected Neural Networks](Assignment/A3/fully_connected_networks.ipynb): This is an extension of the Two-Layer Nets, featuring multiple linear layers and activation functions. In this assignment, you will also explore different gradient descent strategies such as SGD, SGD with Momentum, RMSProp, and Adam to accelerate training and optimize performance. Additionally, you will learn about another regularization technique called Dropout, which behaves differently during training and testing. The network will again be trained on the CIFAR-10 dataset. For reference on gradient descent strategies, see here
, and for Dropout, see here

- [Convolutional Neural Networks](Assignment/A3/convolutional_networks.ipynb): This assignment will walk you through the implementation of **convolutional layer, Kaiming Initialization, pooling layer and batch normalization both in 1D and 2D**. Finally you will build up your own deep CNN network on your own without the aid of PyTorch modules. You should see a huge progress in the accuracy of Image Classification trained on CIFAR-10 dataset due to the powerful CNN architecture and know why CNN is far better than just MLP. 

### A5

- [Recurrent Neural Network](Assignment/A5/rnn_lstm_captioning.ipynb): In this assignment, you will implement Vanilla RNN, LSTM, and Attention LSTM to generate natural language captions for images using the COCO Captions dataset.
  
- [Transformer](Assignment/A5/Transformer.ipynb): Implement a simplified Encoder-Decoder Transformer for sequence-to-sequence tasks, demonstrated on fixed-length arithmetic expressions. The assignment also includes visualizations to help understand attention mechanisms and trains the model end-to-end.

## Acknowledgements
This project references and was inspired by [yzhbradoodrrpurp/EECS498](https://github.com/yzhbradoodrrpurp/EECS498). Thanks for the helpful guidance and examples.

