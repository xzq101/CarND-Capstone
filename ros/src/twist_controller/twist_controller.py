import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
	self.max_abs_angle = kwargs.get('max_steer_angle')  
        self.fuel_capacity = kwargs.get('fuel_capacity')
        self.vehicle_mass = kwargs.get('vehicle_mass')
        self.wheel_radius = kwargs.get('wheel_radius')
        self.accel_limit = kwargs.get('accel_limit')
        self.decel_limit = kwargs.get('decel_limit')
        self.brake_deadband = kwargs.get('brake_deadband')
        self.max_acceleration = 1.5
        self.wheel_base = kwargs.get('wheel_base')
        self.steer_ratio = kwargs.get('steer_ratio')
        self.max_steer_angle = kwargs.get('max_steer_angle')
        self.max_lat_accel = kwargs.get('max_lat_accel')

        self.yaw_controller = YawController(
            self.wheel_base, self.steer_ratio, 0.1,
            self.max_lat_accel, self.max_steer_angle) 

        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # Minimum throttle value
        mx = 0.2 # Max throttle value        

        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = .02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)
	#self.brake_lpf = LowPassFilter(0.5, 0.02)


        #total_vehicle_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY # 1.774,933
        # max torque (1.0 throttle) and  max brake torque (deceleration lmt)
        #self.max_acc_torque = total_vehicle_mass * self.max_acceleration * self.wheel_radius #567
        #self.max_brake_torque = total_vehicle_mass * abs(self.decel_limit) * self.wheel_radius #2095
        
        self.last_time = rospy.get_time()

 
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
      
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400 # N*m - to hold the car in place if we are stopped at a light.  Accel ~ 1m/s**2

        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m


	return throttle, brake, steering


