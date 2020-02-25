# libtorchmnist

Pytorch C++ frontend implementation to recognize handwritten digits.

## Contents

* [What is LIBTORCHMNIST?](#what-is-libtorchmnist)
* [Project Hierarchy](#project-hierarchy)
* [Getting Started](#getting-started)
* [Testing](#testing)
* [Support](#support)
* [FAQ](#faq)
* [Development](#development)
* [License](#license)
* [References](#references)

## What is LIBTORCHMNIST?

Example project for recognition of the Modified National Institute of Standards and Technology database of handwritten digits

[Contents](#contents)


## Project Hierarchy

```console
.
├── CMakeLists.txt
├── download_mnist.py
├── LICENSE
├── README.md
└── source
    ├── applications
    │   ├── CMakeLists.txt
    │   └── mnist
    │       ├── CMakeLists.txt
    │       ├── main.cpp
    │       ├── main.h
    │       └── main.h.in
    ├── CMakeLists.txt
    └── libraries
        ├── CMakeLists.txt
        └── DigitsRecognition
            ├── CMakeLists.txt
            └── TorchImplementation
                ├── CMakeLists.txt
                ├── TorchNetwork.cpp
                ├── TorchNetwork.h
                └── TorchNetwork_test.cpp
```

[Contents](#contents)


## Getting Started

```console
libtorchmnist$ mkdir build
libtorchmnist$ cd build
libtorchmnist/build$ cmake -DCMAKE_PREFIX_PATH=</path/to/libtorch> .. && make
```

[Contents](#contents)


## Testing

Contents](#contents)


## Support

Technical support is available in 

[Contents](#contents)

## FAQ

[Contents](#contents)


## Development

If you want to contribute:

### Process

1. Review the Contribution License Agreement
   1.1. Defines the terms under which intellectual property is contributed to a project
   1.2. To ensure the owner of the project has the necessary ownership, or grants of rights over contributions made by third parties.
   1.3. Must be signed by any contributor to a project who is making a "significant contribution".
   1.4. Without that signed CLA the contribution will not be accepted.
2. Check the project issues list
3. Fork the repository
4. Submit your contribution as a pull request

### Discuss your Change

- Create/update an issue
- A committer approves the proposal

### Making changes

Making a copy of the project's code
Make your code changes
Keep your changes concise and easy for the maintainers to understand
Refer back to any discussion you had on the projects issues list
Stick to the scope of the issue
Make sure to fully test your code
Make sure to include any required documentation
Remember your contributions represent your public reputation and that of DXC

### Submitting Changes

- Request review
- Create a contribution request (Pull Request) which include:
  - Your code changes
  - A commit message describing what you're submitting
  - A reference to the original issue you created/updated in step 1  

[Contents](#contents)


## License

See [LICENSE](LICENSE) file

[Contents](#contents)


## References

1. [MNIST Example with the PyTorch C++ Frontend](https://github.com/pytorch/examples/tree/master/cpp/mnist)

[Contents](#contents)

