from flight_simulator.core.vehicle.models.equations_of_motion.example_phase_info import phase_info
import csdl_alpha as csdl
import numpy as np
from flight_simulator.core.loads.mass_properties import MassProperties
from flight_simulator.core.dynamics.aircraft_states import AircraftStates
from typing import Union
from dataclasses import dataclass
import aviary.api as av

class AviaryAnalysis:
    def __init__(self, aircraft_data_file_path: str, aircraft_data_file_name: str):
        """
        Runs the Aviary analysis for each phase of mission
        Can be just one or multiple phases


        """

        aircraft_data = aircraft_data_file_path / aircraft_data_file_name

        prob = av.AviaryProblem()
        prob.load_inputs(aircraft_data, phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP", max_iter=100)

        prob.add_design_variables()

        # Load optimization problem formulation
        # Detail which variables the optimizer can control
        prob.add_objective()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_aviary_problem()