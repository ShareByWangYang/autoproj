#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

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
        
        double theta_d = theta + k1 * theta * theta * theta +
                         k2 * theta * theta * theta * theta * theta +
                         k3 * theta * theta * theta * theta * theta * theta * theta +
                         k4 * theta * theta * theta * theta * theta * theta * theta * theta * theta;
        
        double safe_r = std::max(r, 1e-10);
        double scale = theta_d / safe_r;
        
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
    py::buffer_info points_info = points_3d.request();
    py::buffer_info dist_info = dist_coeffs.request();
    
    double* points_ptr = static_cast<double*>(points_info.ptr);
    double* dist_ptr = static_cast<double*>(dist_info.ptr);
    
    int n_points = points_info.shape[0];
    
    py::array_t<double> pixels({n_points, 2});
    py::array_t<bool> valid({n_points});
    
    py::buffer_info pixels_info = pixels.request();
    py::buffer_info valid_info = valid.request();
    
    double* pixels_ptr = static_cast<double*>(pixels_info.ptr);
    bool* valid_ptr = static_cast<bool*>(valid_info.ptr);
    
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0;
    if (dist_info.shape[0] >= 4) {
        k1 = dist_ptr[0]; k2 = dist_ptr[1]; p1 = dist_ptr[2]; p2 = dist_ptr[3];
    }
    if (dist_info.shape[0] >= 8) {
        k3 = dist_ptr[4]; k4 = dist_ptr[5]; k5 = dist_ptr[6]; k6 = dist_ptr[7];
    }
    
    project_pinhole(
        points_ptr, pixels_ptr, valid_ptr, n_points,
        fx, fy, cx, cy,
        k1, k2, p1, p2, k3, k4, k5, k6,
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
    py::buffer_info points_info = points_3d.request();
    double* points_ptr = static_cast<double*>(points_info.ptr);
    int n_points = points_info.shape[0];
    
    py::array_t<double> pixels({n_points, 2});
    py::array_t<bool> valid({n_points});
    
    py::buffer_info pixels_info = pixels.request();
    py::buffer_info valid_info = valid.request();
    
    double* pixels_ptr = static_cast<double*>(pixels_info.ptr);
    bool* valid_ptr = static_cast<bool*>(valid_info.ptr);
    
    project_kannala_brandt(
        points_ptr, pixels_ptr, valid_ptr, n_points,
        fx, fy, cx, cy, k1, k2, k3, k4,
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
    py::buffer_info points_info = points_3d.request();
    py::buffer_info poly_info = fw_poly.request();
    
    double* points_ptr = static_cast<double*>(points_info.ptr);
    double* poly_ptr = static_cast<double*>(poly_info.ptr);
    
    int n_points = points_info.shape[0];
    int poly_size = poly_info.shape[0];
    
    py::array_t<double> pixels({n_points, 2});
    py::array_t<bool> valid({n_points});
    
    py::buffer_info pixels_info = pixels.request();
    py::buffer_info valid_info = valid.request();
    
    double* pixels_ptr = static_cast<double*>(pixels_info.ptr);
    bool* valid_ptr = static_cast<bool*>(valid_info.ptr);
    
    project_ftheta(
        points_ptr, pixels_ptr, valid_ptr, n_points,
        poly_ptr, poly_size, cx, cy,
        width, height, near_z, far_z
    );
    
    return py::make_tuple(pixels, valid);
}

PYBIND11_MODULE(_projection_cpp, m) {
    m.doc() = "C++ extension for high-performance 3D-to-2D projection";
    
    m.def("project_pinhole", &py_project_pinhole,
          "Project 3D points using pinhole camera model",
          py::arg("points_3d"), py::arg("fx"), py::arg("fy"),
          py::arg("cx"), py::arg("cy"), py::arg("dist_coeffs"),
          py::arg("width"), py::arg("height"),
          py::arg("near_z"), py::arg("far_z"));
    
    m.def("project_kannala_brandt", &py_project_kannala_brandt,
          "Project 3D points using Kannala-Brandt fisheye model",
          py::arg("points_3d"), py::arg("fx"), py::arg("fy"),
          py::arg("cx"), py::arg("cy"),
          py::arg("k1"), py::arg("k2"), py::arg("k3"), py::arg("k4"),
          py::arg("width"), py::arg("height"),
          py::arg("near_z"), py::arg("far_z"));
    
    m.def("project_ftheta", &py_project_ftheta,
          "Project 3D points using F-Theta fisheye model",
          py::arg("points_3d"), py::arg("fw_poly"),
          py::arg("cx"), py::arg("cy"),
          py::arg("width"), py::arg("height"),
          py::arg("near_z"), py::arg("far_z"));
}