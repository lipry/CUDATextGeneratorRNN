//
// Created by Fabio Lipreri on 2019-05-14.
//

#ifndef PROGETTOGPU_MATRIX_H
#define PROGETTOGPU_MATRIX_H

#include <stdlib.h>
#include <memory>

// Why use Shared pointer
//https://www.quora.com/When-should-I-use-shared_ptr-and-unique_ptr-in-C++-and-what-are-they-good-for

class Matrix {
private:
    size_t x;
    size_t y;

    bool host_alloc;
    bool dev_alloc;

    std::shared_ptr<float> host_data;
    std::shared_ptr<float> dev_data;

    void allocHostMemory();
    void allocDevMemory();
    bool isVector();
public:
    Matrix(size_t x, size_t y);
    Matrix() = default;

    void allocate();
    void allocate_size(size_t x, size_t y);
    bool isDevAlloc();
    bool isHostAlloc();
    void cpyHostToDev();
    void cpyDevToHost();
    void cpyHostToDevCublas();
    void cpyDevToHostCublas();
    void matrix_like(float number, Matrix &x);
    size_t getX() const;
    size_t getY() const;

    void print_matrix();

    const std::shared_ptr<float> &getHostData() const;
    const std::shared_ptr<float> &getDevData() const;

    float& operator[](const int index);
    const float& operator[](const int index) const;
};


#endif //PROGETTOGPU_MATRIX_H
