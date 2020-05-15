#pragma once

#include <stdexcept>
#include <cmath>

template <typename T>
class Matrix
{
    public:
    explicit Matrix(size_t w = 0, size_t h = 0);
    
    T get(size_t i, size_t j) const;
    void set(size_t i, size_t j, const T &data);
    void resize(size_t w, size_t h);
    Matrix<T> get_transposed() const;
    T sum() const;

    size_t width() const noexcept;
    size_t height() const noexcept;

    Matrix<T> operator*(double) const;
    Matrix<T> &operator+=(const Matrix<T> &);
    Matrix<T> &operator-=(const Matrix<T> &);
    Matrix<T> operator-(const Matrix<T> &) const;
    Matrix<T> operator^(T s) const;

    private:
    std::vector<T> mat_;
    size_t w_, h_;
};

template <typename T>
Matrix<T>::Matrix(size_t w, size_t h)
    : mat_(w*h),
      w_ {w},
      h_ {h}
{}

template <typename T>
T Matrix<T>::get(size_t i, size_t j) const
{
    // ith row, jth column
    if (i > h_ || j > w_)
        throw std::out_of_range("Attempted to access past matrix bounds");
    return mat_[j + i*w_];
}

template <typename T>
void Matrix<T>::set(size_t i, size_t j, const T &data)
{
    if (i > h_ || j > w_)
        throw std::out_of_range("Attempted to access past matrix bounds");
    mat_[j + i*w_] = data;
}

template <typename T>
void Matrix<T>::resize(size_t w, size_t h)
{
    mat_.resize(w*h);
    w_ = w;
    h_ = h;
}

template<typename T>
Matrix<T> Matrix<T>::get_transposed() const
{
    Matrix<T> transposed(h_, w_);
    for (size_t i = 0; i < h_; ++i)
    {
        for (size_t j = 0; j < w_; ++j)
        {
            transposed.set(j, i, get(i, j));
        }
    }
    return transposed;
}

template<typename T>
inline T Matrix<T>::sum() const
{
    T sum { 0 };
    for (const T &x : mat_)
        sum += x;
    return sum;
}

template <typename T>
inline size_t Matrix<T>::width() const noexcept
{
    return w_;
}

template <typename T>
inline size_t Matrix<T>::height() const noexcept
{
    return h_;
}

template<typename T>
inline Matrix<T> Matrix<T>::operator*(double s) const
{
    Matrix<T> out(w_, h_);
    for (size_t i = 0; i < w_ * h_; ++i)
    {
        out.mat_[i] = this->mat_[i] * s;
    }
    return out;
}

template<typename T>
inline Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &b)
{
    if (w_ != b.width() || h_ != b.height())
        throw std::runtime_error("Can't subtract matrices with different \
            dimensions");
    for (size_t i = 0; i < w_ * h_; ++i)
        mat_[i] += b.mat_[i];
    return *this;
}

template<typename T>
inline Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &b)
{
    if (w_ != b.width() || h_ != b.height())
        throw std::runtime_error("Can't subtract matrices with different \
            dimensions");
    for (size_t i = 0; i < w_ * h_; ++i)
        mat_[i] -= b.mat_[i];
    return *this;
}

template<typename T>
inline Matrix<T> Matrix<T>::operator-(const Matrix<T> &m) const
{
    if (m.width() != w_ || m.height() != h_)
        throw std::runtime_error("Can't subtract different sized matrices");
    Matrix<T> diff(*this);
    diff -= m;
    return diff;
}

template<typename T>
inline Matrix<T> Matrix<T>::operator^(T s) const
{
    Matrix<T> out(w_, h_);
    for (size_t i { 0 }; i < w_ * h_; ++i)
    {
        out.mat_[i] = std::pow(mat_[i], s);
    }
    return out;
}
