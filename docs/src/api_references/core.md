# Core

## Vehicle Components

### Component Base Class

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.component.Component
    :members:
    :special-members:
```

### Aircraft

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.aircraft.Aircraft
    :members:
    :special-members:
```

### Wing

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.wing.WingParameters
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.wing.WingGeometricQuantities
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.wing.Wing
    :members:
    :special-members:
```

### Fuselage

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.fuselage.FuselageParameters
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.fuselage.FuselageGeometricQuantities
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.fuselage.Fuselage
    :members:
    :special-members:
```

### Rotor

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.rotor.RotorParameters
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.rotor.Rotor
    :members:
    :special-members:
```

### Powertrain

```{eval-rst}
.. autoclass:: falco.core.vehicle.components.powertrain.Powertrain
    :members:
    :special-members:
```

## Vehicle Conditions

### Condition Base Class

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.Condition
    :members:
    :special-members:
```

### Flight Condition Parameters

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.HoverParameters
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.ClimbParameters
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.CruiseParameters
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.RateofClimbParameters
    :members:
    :special-members:
```

### Flight Conditions

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.HoverCondition
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.ClimbCondition
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.CruiseCondition
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.conditions.aircraft_conditions.RateofClimb
    :members:
    :special-members:
```

## Vehicle Controls

```{eval-rst}
.. autoclass:: falco.core.vehicle.controls.vehicle_control_system.ControlSurface
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.controls.vehicle_control_system.PropulsiveControl
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.controls.vehicle_control_system.PropulsiveControlRPM
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.vehicle.controls.vehicle_control_system.VehicleControlSystem
    :members:
    :special-members:
```

## Dynamics

### Vector

```{eval-rst}
.. autoclass:: falco.core.dynamics.vector.Vector
    :members:
    :special-members:
```

### Axis System

```{eval-rst}
.. autoclass:: falco.core.dynamics.axis.Axis
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.axis.ValidOrigins
    :members:
    :special-members:
```

### Aircraft States

```{eval-rst}
.. autoclass:: falco.core.dynamics.aircraft_states.RigidBodyStates
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.aircraft_states.MassSpringDamperState
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.aircraft_states.AircraftStates
    :members:
    :special-members:
```

### Equations of Motion

```{eval-rst}
.. autoclass:: falco.core.dynamics.EoM.StateVectorDot
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.EoM.DynamicSystem
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.EoM.EquationsOfMotion
    :members:
    :special-members:
```

### Linear Stability

```{eval-rst}
.. autoclass:: falco.core.dynamics.linear_stability.LinearStabilityMetrics
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.linear_stability.LinearStabilityAnalysis
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.dynamics.linear_stability.EigenValueOperation
    :members:
    :special-members:
```

### Axis System with LSDOGeo

```{eval-rst}
.. autoclass:: falco.core.dynamics.axis_lsdogeo.AxisLsdoGeo
    :members:
    :special-members:
```

## Loads

### Mass Properties

```{eval-rst}
.. autoclass:: falco.core.loads.mass_properties.MassProperties
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.loads.mass_properties.MassMI
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.loads.mass_properties.GravityLoads
    :members:
    :special-members:
```

### Forces and Moments

```{eval-rst}
.. autoclass:: falco.core.loads.forces_moments.Vector
    :members:
    :special-members:
```

```{eval-rst}
.. autoclass:: falco.core.loads.forces_moments.ForcesMoments
    :members:
    :special-members:
```

### Loads

```{eval-rst}
.. autoclass:: falco.core.loads.loads.Loads
    :members:
    :special-members:
```