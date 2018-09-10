#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <cufft.h>
// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// structure used to accumulate the moments and other 
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
    float n;
    float min;
    float max;
    float mean;
    float M2;
    float M3;
    float M4;
    
    // initialize to the identity element
    void initialize()
    {
      n = mean = M2 = M3 = M4 = 0;
      min = std::numeric_limits<T>::max();
      max = std::numeric_limits<T>::min();
    }

    T variance()   { return M2 / (n - 1); }
    T variance_n() { return M2 / n; }
    T skewness()   { return std::sqrt(n) * M3 / std::pow(M2, (T) 1.5); }
    T kurtosis()   { return n * M4 / (M2 * M2); }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
struct summary_stats_unary_op
{
    __host__ __device__
    summary_stats_data<float> operator()(const cufftComplex& val) const
    {
         summary_stats_data<float> result;
         result.n    = 1;
         result.min  = val.x;
         result.max  = val.x;
         result.mean = val.x;
         result.M2   = 0;
         result.M3   = 0;
         result.M4   = 0;

         return result;
    }
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data 
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for 
// all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op 
    : public thrust::binary_function<const summary_stats_data<T>&, 
                                     const summary_stats_data<T>&,
                                           summary_stats_data<T> >
{
    __host__ __device__
    summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
    {
        summary_stats_data<T> result;
        
        // precompute some common subexpressions
        T n  = x.n + y.n;
        T n2 = n  * n;
        T n3 = n2 * n;

        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;
        T delta3 = delta2 * delta;
        T delta4 = delta3 * delta;
        
        //Basic number of samples (n), min, and max
        result.n   = n;
        result.min = thrust::min(x.min, y.min);
        result.max = thrust::max(x.max, y.max);

        result.mean = x.mean + delta * y.n / n;

        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        result.M3  = x.M3 + y.M3;
        result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2; 
        result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;
    
        result.M4  = x.M4 + y.M4;
        result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
        result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
        result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;
        
        return result;
    }
};

// template <typename Iterator>
// void print_range(const std::string& name, Iterator first, Iterator last)
// {
//     typedef typename std::iterator_traits<Iterator>::value_type T;

//     std::cout << name << ": ";
//     thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));  
//     std::cout << "\n";
// }

int main(void)
{
    // typedef cufftComplex T;

    cufftComplex *d_data;
    int N = 1024*1024;

    cufftComplex *h_data = new cufftComplex[N];
    
    for (int jj = 0; jj < N; jj++){
        h_data[jj].x = jj;
        h_data[jj].y = 13.121123410983;
    }
    cudaMalloc(&d_data, N*sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, N*sizeof(cufftComplex), cudaMemcpyHostToDevice);

    thrust::device_ptr<cufftComplex> t_data(d_data);
    // thrust::fill(t_data, t_data + N, 3);


    // initialize host array
    // T h_x[] = {N, 7, 13, 16};

    // transfer to device
    // thrust::device_vector<T> d_x(h_x, h_x + sizeof(h_x) / sizeof(T));

    // setup arguments
    summary_stats_unary_op unary_op;
    summary_stats_binary_op<float> binary_op;
    summary_stats_data<float>      init;

    init.initialize();

    // compute summary statistics
    summary_stats_data<float> result;
    for (int i = 0; i < 20; i++){
        result = thrust::transform_reduce(t_data, t_data + N, unary_op, init, binary_op);
    }

    std::cout <<"******Summary Statistics Example*****"<<std::endl;

    std::cout <<"Count              : "<< result.n << std::endl;
    std::cout <<"Minimum            : "<< result.min <<std::endl;
    std::cout <<"Maximum            : "<< result.max <<std::endl;
    std::cout <<"Mean               : "<< result.mean << std::endl;
    std::cout <<"Variance           : "<< result.variance() << std::endl;
    std::cout <<"Standard Deviation : "<< std::sqrt(result.variance_n()) << std::endl;
    std::cout <<"Skewness           : "<< result.skewness() << std::endl;
    std::cout <<"Kurtosis           : "<< result.kurtosis() << std::endl;

    return 0;
}












