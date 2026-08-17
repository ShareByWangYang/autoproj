#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// OpenMP 并行阈值：仅当点数超过此值时启用多线程，避免小数据集的线程创建开销
#define OMP_THRESHOLD 1000000

void project_pinhole(
    const double* points_3d,
    double* pixels,
    bool* valid,
    int n_points,
    double fx, double fy,
    double cx, double cy,
    double k1, double k2, double p1, double p2,
    double k3, double k4, double k5, double k6,
    int width, int height,
    double near_z, double far_z
) {
    #pragma omp parallel for schedule(static) if(n_points > OMP_THRESHOLD)
    for (int i = 0; i < n_points; ++i) {
        double x_c = points_3d[i * 3];
        double y_c = points_3d[i * 3 + 1];
        double z_c = points_3d[i * 3 + 2];
        
        if (z_c <= near_z || z_c >= far_z) {
            pixels[i * 2] = -1;
            pixels[i * 2 + 1] = -1;
            valid[i] = false;
            continue;
        }
        
        double x_norm = x_c / z_c;
        double y_norm = y_c / z_c;
        
        double r2 = x_norm * x_norm + y_norm * y_norm;
        double r4 = r2 * r2;
        double r6 = r2 * r4;
        
        double numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
        denom = std::max(denom, 1e-10);
        double radial = numerator / denom;
        
        double x_dist = x_norm * radial;
        double y_dist = y_norm * radial;
        
        x_dist += 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm * x_norm);
        y_dist += p1 * (r2 + 2 * y_norm * y_norm) + 2 * p2 * x_norm * y_norm;
        
        double u = fx * x_dist + cx;
        double v = fy * y_dist + cy;
        
        bool in_bounds = (u >= -20) && (u < width + 20) && 
                         (v >= -20) && (v < height + 20);
        
        if (!in_bounds) {
            pixels[i * 2] = -1;
            pixels[i * 2 + 1] = -1;
            valid[i] = false;
        } else {
            // 恢复 clamp 行为，与纯 Python 路径保持一致
            u = std::max(0.0, std::min((double)(width - 1), u));
            v = std::max(0.0, std::min((double)(height - 1), v));
            pixels[i * 2] = u;
            pixels[i * 2 + 1] = v;
            valid[i] = true;
        }
    }
}

void project_kannala_brandt(
    const double* points_3d,
    double* pixels,
    bool* valid,
    int n_points,
    double fx, double fy,
    double cx, double cy,
    double k1, double k2, double k3, double k4,
    int width, int height,
    double near_z, double far_z
) {
    #pragma omp parallel for schedule(static) if(n_points > OMP_THRESHOLD)
    for (int i = 0; i < n_points; ++i) {
        double x_c = points_3d[i * 3];
        double y_c = points_3d[i * 3 + 1];
        double z_c = points_3d[i * 3 + 2];
        
        if (z_c <= near_z || z_c >= far_z) {
            pixels[i * 2] = -1;
            pixels[i * 2 + 1] = -1;
            valid[i] = false;
            continue;
        }
        
        double x_norm = x_c / z_c;
        double y_norm = y_c / z_c;
        
        double r = std::sqrt(x_norm * x_norm + y_norm * y_norm);
        double theta = std::atan(r);
        
        double theta_d = theta + k1 * std::pow(theta, 3) + k2 * std::pow(theta, 5) + 
                         k3 * std::pow(theta, 7) + k4 * std::pow(theta, 9);
        
        double scale = theta_d / std::max(r, 1e-10);
        
        double x_dist = x_norm * scale;
        double y_dist = y_norm * scale;
        
        double u = fx * x_dist + cx;
        double v = fy * y_dist + cy;
        
        bool in_bounds = (u >= -20) && (u < width + 20) && 
                         (v >= -20) && (v < height + 20);
        
        if (!in_bounds) {
            pixels[i * 2] = -1;
            pixels[i * 2 + 1] = -1;
            valid[i] = false;
        } else {
            // 恢复 clamp 行为，与纯 Python 路径保持一致
            u = std::max(0.0, std::min((double)(width - 1), u));
            v = std::max(0.0, std::min((double)(height - 1), v));
            pixels[i * 2] = u;
            pixels[i * 2 + 1] = v;
            valid[i] = true;
        }
    }
}

void project_ftheta(
    const double* points_3d,
    double* pixels,
    bool* valid,
    int n_points,
    const double* fw_poly,
    int poly_size,
    double cx, double cy,
    int width, int height,
    double near_z, double far_z
) {
    #pragma omp parallel for schedule(static) if(n_points > OMP_THRESHOLD)
    for (int i = 0; i < n_points; ++i) {
        double x_c = points_3d[i * 3];
        double y_c = points_3d[i * 3 + 1];
        double z_c = points_3d[i * 3 + 2];
        
        if (z_c <= near_z || z_c >= far_z) {
            pixels[i * 2] = -1;
            pixels[i * 2 + 1] = -1;
            valid[i] = false;
            continue;
        }
        
        double theta = std::atan2(std::sqrt(x_c * x_c + y_c * y_c), z_c);
        
        double r = 0.0;
        double theta_pow = 1.0;
        for (int j = 0; j < poly_size; ++j) {
            r += fw_poly[j] * theta_pow;
            theta_pow *= theta;
        }
        
        double phi = std::atan2(y_c, x_c);
        
        double u = r * std::cos(phi) + cx;
        double v = r * std::sin(phi) + cy;
        
        bool in_bounds = (u >= -20) && (u < width + 20) && 
                         (v >= -20) && (v < height + 20);
        
        if (!in_bounds) {
            pixels[i * 2] = -1;
            pixels[i * 2 + 1] = -1;
            valid[i] = false;
        } else {
            // 恢复 clamp 行为，与纯 Python 路径保持一致
            u = std::max(0.0, std::min((double)(width - 1), u));
            v = std::max(0.0, std::min((double)(height - 1), v));
            pixels[i * 2] = u;
            pixels[i * 2 + 1] = v;
            valid[i] = true;
        }
    }
}

py::tuple py_project_pinhole(
    py::array_t<double> points_3d,
    double fx, double fy,
    double cx, double cy,
    py::array_t<double> dist_coeffs,
    int width, int height,
    double near_z, double far_z
) {
    py::buffer_info buf = points_3d.request();
    int n_points = buf.shape[0];
    
    py::array_t<double> pixels({n_points, 2});
    py::array_t<bool> valid(n_points);
    
    py::buffer_info dist_buf = dist_coeffs.request();
    int dist_size = dist_buf.shape[0];
    double* dist_ptr = (double*)dist_buf.ptr;
    
    double k1 = dist_size > 0 ? dist_ptr[0] : 0.0;
    double k2 = dist_size > 1 ? dist_ptr[1] : 0.0;
    double p1 = dist_size > 2 ? dist_ptr[2] : 0.0;
    double p2 = dist_size > 3 ? dist_ptr[3] : 0.0;
    double k3 = dist_size > 4 ? dist_ptr[4] : 0.0;
    double k4 = dist_size > 5 ? dist_ptr[5] : 0.0;
    double k5 = dist_size > 6 ? dist_ptr[6] : 0.0;
    double k6 = dist_size > 7 ? dist_ptr[7] : 0.0;
    
    project_pinhole(
        (double*)buf.ptr, (double*)pixels.request().ptr, (bool*)valid.request().ptr,
        n_points, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6,
        width, height, near_z, far_z
    );
    
    return py::make_tuple(pixels, valid);
}

py::tuple py_project_kannala_brandt(
    py::array_t<double> points_3d,
    double fx, double fy,
    double cx, double cy,
    double k1, double k2, double k3, double k4,
    int width, int height,
    double near_z, double far_z
) {
    py::buffer_info buf = points_3d.request();
    int n_points = buf.shape[0];
    
    py::array_t<double> pixels({n_points, 2});
    py::array_t<bool> valid(n_points);
    
    project_kannala_brandt(
        (double*)buf.ptr, (double*)pixels.request().ptr, (bool*)valid.request().ptr,
        n_points, fx, fy, cx, cy, k1, k2, k3, k4,
        width, height, near_z, far_z
    );
    
    return py::make_tuple(pixels, valid);
}

py::tuple py_project_ftheta(
    py::array_t<double> points_3d,
    py::array_t<double> fw_poly,
    double cx, double cy,
    int width, int height,
    double near_z, double far_z
) {
    py::buffer_info buf = points_3d.request();
    int n_points = buf.shape[0];
    
    py::array_t<double> pixels({n_points, 2});
    py::array_t<bool> valid(n_points);
    
    py::buffer_info poly_buf = fw_poly.request();
    int poly_size = poly_buf.shape[0];
    
    project_ftheta(
        (double*)buf.ptr, (double*)pixels.request().ptr, (bool*)valid.request().ptr,
        n_points, (double*)poly_buf.ptr, poly_size,
        cx, cy, width, height, near_z, far_z
    );
    
    return py::make_tuple(pixels, valid);
}

PYBIND11_MODULE(_projection_cpp, m) {
    m.def("project_pinhole", &py_project_pinhole,
          "Project 3D points to 2D using pinhole camera model");
    m.def("project_kannala_brandt", &py_project_kannala_brandt,
          "Project 3D points to 2D using Kannala-Brandt fisheye model");
    m.def("project_ftheta", &py_project_ftheta,
          "Project 3D points to 2D using F-Theta fisheye model");
}
