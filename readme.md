Micrograd Autograd Engine (Minimal)
This project contains a minimal implementation of a reverse-mode automatic differentiation engine, similar to what is used in deep learning libraries like PyTorch. It focuses on the core building block: a scalar value and its associated gradient, with support for backpropagation through elementary operations.

Class: Value
The Value class represents a single scalar value used in computation. It tracks its own gradient and constructs a computational graph to enable automatic differentiation.

Features
Basic arithmetic operations: +, -, *, /, **

Activation function: ReLU

Gradient computation via .backward()

Operator overloading for intuitive usage

Simple topological sorting for correct backpropagation order

How It Works

Each time you perform an operation (like addition or multiplication), a new Value object is created, and a computation graph is built. When .backward() is called on the final output, it uses the chain rule to calculate gradients of all contributing inputs via reverse-mode autodiff.

Methods Overview
__add__, __mul__, __pow__, __truediv__, etc.: Support basic mathematical operations

relu(): Applies the ReLU activation function

backward(): Triggers backpropagation to compute gradients

__repr__(): Returns a string representation for easy debugging

Internal Attributes
_prev: Tracks parent nodes to build the computation graph

_backward: Function that applies the gradient rules

grad: Stores the gradient with respect to the final output

Getting Started
You can directly use the Value class in your projects to perform scalar-based differentiation for educational or experimental purposes.