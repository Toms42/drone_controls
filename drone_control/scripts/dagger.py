import numpy as np
from train_env import Environment, Expert
from BCModel import BCModel

def train_dagger():
    # network parameters
    log_name "../logs"
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
    agent = DroneAgent(log_name, learning_rate, batch_size, num_epochs)

    # define training data
    expert_data = {
        'observations': []
        'actions': []
    }
    
    env = Environment(aggr=1)
    expert = Expert(env.x0, env.dt, env.m)
    Kp_vals = np.arange(start=5, stop=10, step=1.25)
    Kd_vals = np.arange(start=0, stop=3, step=1.5)

    # parameters to vary:
    # spline aggressiveness
    # Kp and Kd values for PID
    # Q, R diagonals 

    for kp in Kp_vals:
        for kd in Kd_vals:
            expert.change_pids(
                phi_params=[kp, 0, kd],
                theta_params=[kp, 0, kd]
            )
            print("Kp: %.3f, Kd: %.3f" % (kp, kd))
            env.start(Nsecs=10)
            while env.sim_running:
                try:
                    # feedforward acceleration
                    ff = env.get_feedfwd()

                    # get target pose
                    xref = env.get_xref()
                    
                    # get current state
                    (trans, rot) = env.get_agent_pose()
                except Exception as e:
                    print(e)
                    continue

                action = expert.gen_action((trans, rot), xref, ff)
                env.step(action)
                env.rate.sleep()

class DroneAgent(BCModel):
    def __init__(self, log_name, learning_rate, batch_size, num_epochs):
        self.sym_state = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw']
        self.sym_action = ['thrust', 'dphi', 'dtheta', 'dpsi']
        observation_dim = (len(self.sym_state)*2, 1)  # current and target position
        action_dim = (len(self.sym_action), 1)
        super().__init__(log_name=log_name, observation_dim=observation_dim, action_dim=action_dim,
            learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)

    def 



