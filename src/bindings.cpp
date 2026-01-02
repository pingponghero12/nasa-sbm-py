#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <array>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// ESA NASA SBM headers
#include "breakupModel/model/Satellite.h"
#include "breakupModel/model/SatelliteBuilder.h"
#include "breakupModel/simulation/Breakup.h"
#include "breakupModel/simulation/Explosion.h"
#include "breakupModel/simulation/Collision.h"
#include "breakupModel/input/RuntimeInputSource.h"
#include "breakupModel/simulation/BreakupBuilder.h"

namespace py = pybind11;

py:: array_t<double> array_to_numpy(const std::array<double, 3>& arr) {
    return py::array_t<double>(3, arr.data());
}

// Helper function to calculate circular orbital velocity
std::array<double, 3> calculate_circular_velocity(double altitude_km) {
    const double mu = 398600.4418e9;  // m^3/s^2
    const double R_earth = 6378137.0;  // m
    double r = R_earth + altitude_km * 1000.0;
    double v_circular = std::sqrt(mu / r);
    return {v_circular, 0.0, 0.0};
}

// Helper function to calculate position vector
std::array<double, 3> calculate_position(double altitude_km) {
    const double R_earth = 6378137.0;  // m
    double r = R_earth + altitude_km * 1000.0;
    return {r, 0.0, 0.0};
}

class BreakupSimulation {
private: 
    std::unique_ptr<Breakup> breakup_;
    std::vector<Satellite> results_;
    double orbit_altitude_;

public:
    BreakupSimulation() : orbit_altitude_(0.0) {}

    void run_explosion(
        double mass,
        const std::string& sat_type_str,
        double min_lc,
        std::optional<int> seed = std::nullopt,
        bool enforce_mass_conservation = false,
        std::optional<double> orbit_altitude = std::nullopt,
        std::optional<std::array<double, 3>> position = std::nullopt,
        std::optional<std:: array<double, 3>> velocity = std::nullopt
    ) {
        SatelliteBuilder builder;
        builder.setID(1)
               .setName("Parent")
               .setMass(mass);

        // Set position and velocity
        if (position.has_value() && velocity.has_value()) {
            // User provided both - use them directly (in meters and m/s)
            builder.setPosition(position.value())
                   .setVelocity(velocity.value());
            orbit_altitude_ = 0.0;  // Not applicable
        } else if (orbit_altitude.has_value()) {
            // Calculate from altitude (circular orbit assumption)
            orbit_altitude_ = orbit_altitude.value();
            builder.setPosition(calculate_position(orbit_altitude_))
                   .setVelocity(calculate_circular_velocity(orbit_altitude_));
        } else {
            // Default:  zero position and velocity
            builder.setPosition({0.0, 0.0, 0.0})
                   .setVelocity({0.0, 0.0, 0.0});
            orbit_altitude_ = 0.0;
        }

        if (sat_type_str == "spacecraft" || sat_type_str == "SC") {
            builder.setSatType(SatType::SPACECRAFT);
        } else if (sat_type_str == "rocket_body" || sat_type_str == "RB") {
            builder.setSatType(SatType::ROCKET_BODY);
        } else {
            builder.setSatType(SatType::SPACECRAFT);
        }

        Satellite sat = builder.getResult();
        std::vector<Satellite> input = {sat};

        breakup_ = std::make_unique<Explosion>(
            input, min_lc, 1, enforce_mass_conservation
        );

        if (seed.has_value()) {
            breakup_->setSeed(*seed);
        }

        breakup_->run();
        results_ = breakup_->getResult();
    }

    void run_collision(
        double mass1,
        double mass2,
        double velocity_relative,
        const std::string& sat_type1_str,
        const std::string& sat_type2_str,
        double min_lc,
        std::optional<int> seed = std::nullopt,
        bool enforce_mass_conservation = false,
        std::optional<double> orbit_altitude = std::nullopt,
        std::optional<std:: array<double, 3>> position = std::nullopt,
        std::optional<std::array<double, 3>> velocity1 = std::nullopt,
        std::optional<std::array<double, 3>> velocity2 = std::nullopt
    ) {
        SatelliteBuilder builder1;
        builder1.setID(1)
                .setName("Satellite1")
                .setMass(mass1);

        SatelliteBuilder builder2;
        builder2.setID(2)
                .setName("Satellite2")
                .setMass(mass2);

        // Set positions and velocities
        if (position.has_value() && velocity1.has_value() && velocity2.has_value()) {
            // User provided everything - use directly (in meters and m/s)
            builder1.setPosition(position.value())
                    .setVelocity(velocity1.value());
            builder2.setPosition(position.value())
                    .setVelocity(velocity2.value());
            orbit_altitude_ = 0.0;
        } else if (orbit_altitude.has_value()) {
            // Calculate from altitude (circular orbit + relative velocity)
            orbit_altitude_ = orbit_altitude.value();
            auto pos = calculate_position(orbit_altitude_);
            auto v_circular = calculate_circular_velocity(orbit_altitude_);
            
            builder1.setPosition(pos)
                    .setVelocity({v_circular[0] + velocity_relative * 1000.0, 
                                  v_circular[1], 
                                  v_circular[2]});
            builder2.setPosition(pos)
                    .setVelocity(v_circular);
        } else {
            // Default: velocity_relative is relative velocity in km/s
            builder1.setPosition({0.0, 0.0, 0.0})
                    .setVelocity({velocity_relative * 1000.0, 0.0, 0.0});
            builder2.setPosition({0.0, 0.0, 0.0})
                    .setVelocity({0.0, 0.0, 0.0});
            orbit_altitude_ = 0.0;
        }

        if (sat_type1_str == "spacecraft" || sat_type1_str == "SC") {
            builder1.setSatType(SatType:: SPACECRAFT);
        } else if (sat_type1_str == "rocket_body" || sat_type1_str == "RB") {
            builder1.setSatType(SatType::ROCKET_BODY);
        }

        if (sat_type2_str == "spacecraft" || sat_type2_str == "SC") {
            builder2.setSatType(SatType::SPACECRAFT);
        } else if (sat_type2_str == "rocket_body" || sat_type2_str == "RB") {
            builder2.setSatType(SatType::ROCKET_BODY);
        }

        Satellite sat1 = builder1.getResult();
        Satellite sat2 = builder2.getResult();
        std::vector<Satellite> input = {sat1, sat2};

        breakup_ = std::make_unique<Collision>(
            input, min_lc, 2, enforce_mass_conservation
        );

        if (seed.has_value()) {
            breakup_->setSeed(*seed);
        }

        breakup_->run();
        results_ = breakup_->getResult();
    }

    size_t get_fragment_count() const {
        return results_.size();
    }

    double get_orbit_altitude() const {
        return orbit_altitude_;
    }

    py::dict get_fragments() {
        py::dict data;
        size_t n = results_.size();

        std::vector<double> sizes;
        std::vector<double> masses;
        std::vector<double> areas;
        std:: vector<double> area_to_mass_ratios;

        sizes.reserve(n);
        masses.reserve(n);
        areas.reserve(n);
        area_to_mass_ratios.reserve(n);

        py::array_t<double> vel_array({static_cast<py:: ssize_t>(n), static_cast<py::ssize_t>(3)});
        py::array_t<double> ejection_vel_array({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
        py::array_t<double> pos_array({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});

        auto vel_buf = vel_array.mutable_unchecked<2>();
        auto ejection_vel_buf = ejection_vel_array.mutable_unchecked<2>();
        auto pos_buf = pos_array.mutable_unchecked<2>();

        for (size_t i = 0; i < n; ++i) {
            const auto& sat = results_[i];
            sizes.push_back(sat.getCharacteristicLength());
            masses.push_back(sat.getMass());
            areas.push_back(sat.getArea());
            area_to_mass_ratios.push_back(sat.getAreaToMassRatio());

            auto v = sat.getVelocity();
            auto ev = sat.getEjectionVelocity();
            auto p = sat.getPosition();
            
            for (size_t j = 0; j < 3; ++j) {
                vel_buf(i, j) = v[j] / 1000.0;  // Convert m/s to km/s
                ejection_vel_buf(i, j) = ev[j] / 1000.0;  // Convert m/s to km/s
                pos_buf(i, j) = p[j] / 1000.0;  // Convert m to km
            }
        }

        data["characteristic_length"] = py::array_t<double>(n, sizes.data());
        data["mass"] = py::array_t<double>(n, masses.data());
        data["area"] = py::array_t<double>(n, areas.data());
        data["area_to_mass_ratio"] = py::array_t<double>(n, area_to_mass_ratios.data());
        data["velocity"] = vel_array;
        data["ejection_velocity"] = ejection_vel_array;
        data["position"] = pos_array;

        return data;
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for ESA NASA Standard Breakup Model";
    
    py::class_<BreakupSimulation>(m, "BreakupSimulation")
        .def(py::init<>())
        .def("run_explosion", &BreakupSimulation::run_explosion,
             py::arg("mass"),
             py::arg("sat_type"),
             py::arg("min_lc"),
             py::arg("seed") = py::none(),
             py::arg("enforce_mass_conservation") = false,
             py::arg("orbit_altitude") = py::none(),
             py::arg("position") = py::none(),
             py::arg("velocity") = py::none())
        .def("run_collision", &BreakupSimulation::run_collision,
             py::arg("mass1"),
             py::arg("mass2"),
             py::arg("velocity_relative"),
             py::arg("sat_type1"),
             py::arg("sat_type2"),
             py::arg("min_lc"),
             py::arg("seed") = py::none(),
             py::arg("enforce_mass_conservation") = false,
             py::arg("orbit_altitude") = py::none(),
             py::arg("position") = py::none(),
             py::arg("velocity1") = py::none(),
             py::arg("velocity2") = py::none())
        .def("get_fragment_count", &BreakupSimulation::get_fragment_count)
        .def("get_fragments", &BreakupSimulation::get_fragments)
        .def("get_orbit_altitude", &BreakupSimulation::get_orbit_altitude);

    m.attr("__version__") = "0.1.0";
}
