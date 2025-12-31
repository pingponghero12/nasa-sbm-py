#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <array>

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

py::array_t<double> array_to_numpy(const std::array<double, 3>& arr) {
    return py::array_t<double>(3, arr.data());
}

class BreakupSimulation {
private:
    std::unique_ptr<Breakup> breakup_;
    std::vector<Satellite> results_;

public:
    BreakupSimulation() = default;

    void run_explosion(
        double mass,
        const std::string& sat_type_str,
        double min_lc,
        std::optional<int> seed = std::nullopt,
        bool enforce_mass_conservation = false
    ) {
        SatelliteBuilder builder;
        builder.setID(1)
               .setName("Parent")
               .setMass(mass)
               .setVelocity({0.0, 0.0, 0.0})
               .setPosition({0.0, 0.0, 0.0});

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
        double velocity,
        const std::string& sat_type1_str,
        const std::string& sat_type2_str,
        double min_lc,
        std::optional<int> seed = std::nullopt,
        bool enforce_mass_conservation = false
    ) {
        SatelliteBuilder builder1;
        builder1.setID(1)
                .setName("Satellite1")
                .setMass(mass1)
                .setVelocity({velocity * 1000.0, 0.0, 0.0})
                .setPosition({0.0, 0.0, 0.0});

        if (sat_type1_str == "spacecraft" || sat_type1_str == "SC") {
            builder1.setSatType(SatType::SPACECRAFT);
        } else if (sat_type1_str == "rocket_body" || sat_type1_str == "RB") {
            builder1.setSatType(SatType::ROCKET_BODY);
        }

        Satellite sat1 = builder1.getResult();

        SatelliteBuilder builder2;
        builder2.setID(2)
                .setName("Satellite2")
                .setMass(mass2)
                .setVelocity({0.0, 0.0, 0.0})
                .setPosition({0.0, 0.0, 0.0});

        if (sat_type2_str == "spacecraft" || sat_type2_str == "SC") {
            builder2.setSatType(SatType::SPACECRAFT);
        } else if (sat_type2_str == "rocket_body" || sat_type2_str == "RB") {
            builder2.setSatType(SatType::ROCKET_BODY);
        }

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

    py::dict get_fragments() {
        py::dict data;
        size_t n = results_.size();

        std::vector<double> sizes;
        std::vector<double> masses;
        std::vector<double> areas;
        std::vector<double> area_to_mass_ratios;

        sizes.reserve(n);
        masses.reserve(n);
        areas.reserve(n);
        area_to_mass_ratios.reserve(n);

        py::array_t<double> vel_array({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
        py::array_t<double> pos_array({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});

        auto vel_buf = vel_array.mutable_unchecked<2>();
        auto pos_buf = pos_array.mutable_unchecked<2>();

        for (size_t i = 0; i < n; ++i) {
            const auto& sat = results_[i];
            sizes.push_back(sat.getCharacteristicLength());
            masses.push_back(sat.getMass());
            areas.push_back(sat.getArea());
            area_to_mass_ratios.push_back(sat.getAreaToMassRatio());

            auto v = sat.getVelocity();
            auto p = sat.getPosition();
            for (size_t j = 0; j < 3; ++j) {
                vel_buf(i, j) = v[j] / 1000.0;
                pos_buf(i, j) = p[j] / 1000.0;
            }
        }

        data["characteristic_length"] = py::array_t<double>(n, sizes.data());
        data["mass"] = py::array_t<double>(n, masses.data());
        data["area"] = py::array_t<double>(n, areas.data());
        data["area_to_mass_ratio"] = py::array_t<double>(n, area_to_mass_ratios.data());
        data["velocity"] = vel_array;
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
             py::arg("enforce_mass_conservation") = false)
        .def("run_collision", &BreakupSimulation::run_collision,
             py::arg("mass1"),
             py::arg("mass2"),
             py::arg("velocity"),
             py::arg("sat_type1"),
             py::arg("sat_type2"),
             py::arg("min_lc"),
             py::arg("seed") = py::none(),
             py::arg("enforce_mass_conservation") = false)
        .def("get_fragment_count", &BreakupSimulation::get_fragment_count)
        .def("get_fragments", &BreakupSimulation::get_fragments);

    m.attr("__version__") = "0.1.0";
}
