from TMC_2209.TMC_2209_StepperDriver import *

tmc = TMC_2209(25, 23, 24)

#tmc.set_direction_reg(False)
tmc.set_current(900)
#tmc.set_interpolation(True)
#tmc.set_spreadcycle(False)
tmc.set_microstepping_resolution(2)
#tmc.set_internal_rsense(False)

#tmc.set_acceleration(2000)
#tmc.set_max_speed(500)

#tmc.set_motor_enabled(True)

#tmc.run_to_position_steps(400)
#tmc.run_to_position_steps(0)

#tmc.set_motor_enabled(False)