import numpy as np

# PID control gains
Kp_x = 0.3  # P-gain for x,y translation error
Kp_yaw = 0.3  # P-gain for yaw error
Ki_x = 0.01  # I-gain for x,y translation error
Ki_yaw = 0.01  # I-gain for yaw error
Kd_x = 0.1  # D-gain for x,y translation error
Kd_yaw = 0.1  # D-gain for yaw error

# PID state
pid_state = {
    "prev_error_x": 0.0,  # Previous error for x
    "prev_error_yaw": 0.0,  # Previous error for yaw
    "integral_x": 0.0,  # Integral of error for x
    "integral_yaw": 0.0,  # Integral of error for yaw
}
# PID limits
pid_limits = {
    "linear": {
        "min": -0.4,  # m/s
        "max": 0.4,  # m/s
    },
    "angular": {
        "min": -0.5,  # rad/s
        "max": 0.5,  # rad/s
    },
}
# PID time step
pid_dt = 0.1  # seconds


def pid_control(
    target_vx,
    target_vyaw,
    current_vx,
    current_vyaw,
    dt=pid_dt,
):
    """
    PID control for velocity command.
    """
    global pid_state

    # Compute errors
    error_x = -target_vx + current_vx
    error_yaw = -target_vyaw + current_vyaw

    # Proportional term
    p_term_x = Kp_x * error_x
    p_term_yaw = Kp_yaw * error_yaw

    # Integral term
    pid_state["integral_x"] += error_x * dt
    pid_state["integral_yaw"] += error_yaw * dt
    i_term_x = Ki_x * pid_state["integral_x"]
    i_term_yaw = Ki_yaw * pid_state["integral_yaw"]

    # Derivative term
    d_term_x = Kd_x * (error_x - pid_state["prev_error_x"]) / dt
    d_term_yaw = Kd_yaw * (error_yaw - pid_state["prev_error_yaw"]) / dt

    # Update previous errors
    pid_state["prev_error_x"] = error_x
    pid_state["prev_error_yaw"] = error_yaw

    # Compute control command
    cmd_linear_x = p_term_x + i_term_x + d_term_x
    cmd_angular_z = p_term_yaw + i_term_yaw + d_term_yaw

    # Apply limits
    cmd_linear_x = np.clip(
        cmd_linear_x, pid_limits["linear"]["min"], pid_limits["linear"]["max"]
    )
    cmd_angular_z = np.clip(
        cmd_angular_z,
        pid_limits["angular"]["min"],
        pid_limits["angular"]["max"],
    )

    return cmd_linear_x, cmd_angular_z
