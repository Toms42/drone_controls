import numpy as np
from train_env import Environment, Expert
from BCModel import BCModel


class DroneAgent(BCModel):
    def __init__(self, log_name, learning_rate, batch_size, num_epochs):
        self.sym_state = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw', 
            'xr', 'yr', 'zr', 'vxr', 'vyr', 'vzr']
        self.sym_action = ['thrust', 'dphi', 'dtheta', 'dpsi']
        observation_dim = len(self.sym_state)  # current pose, target lin pos/vel
        action_dim = len(self.sym_action) 
        super(DroneAgent, self).__init__(log_name=log_name, 
            observation_dim=observation_dim, action_dim=action_dim,
            learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)


def run_episode(env, agent, expert, data, use_expert_policy):
    env.start(Nsecs=env.max_time)
    action = None
    while env.sim_running:
        try:
            # feedforward acceleration
            ff = env.get_feedfwd()

            # get target pose
            xref = env.get_xref()
            
            # get current state
            x, rot = env.get_agent_pose()
            state = np.array(x[0:6, 0].tolist() + rot + xref[0:6, 0].tolist())

        except Exception as e:
            print(e)
            continue
        
        # observation from previous action since env step doesn't take effect immediately
        data['observations'].append(state)

        expert_action = np.array(expert.gen_action(x, rot, xref, ff))
        data['actions'].append(expert_action)
        if use_expert_policy:
            env.step(expert_action)
        else:
            print("Expert: %.3f, %.3f, %.3f, %.3f" % tuple(expert_action))
            [agent_action] = agent.infer(state)
            agent_action = np.ndarray.flatten(agent_action).tolist()
            print("Agent: %.3f, %.3f, %.3f, %.3f" % tuple(agent_action))
            env.step(agent_action)

        # environment will update in next iter, which will associate with this current action
        env.rate.sleep()
        

def train_dagger():
    LOAD_FROM_PREV_SESS = True

    # network parameters
    log_name = "../logs"
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
    agent = DroneAgent(log_name, learning_rate, batch_size, num_epochs)
    
    num_expert_episodes = 10
    num_agent_episodes = 5
    target_gates = [0, 1, 2, 3]
    env = Environment(aggr=1, gate_ids=target_gates)
    expert = Expert(env.dt, env.m)

    init_expert_weights = './checkpoint/init_expert.meta'
    final_agent_weights = './checkpoint/final_agent.meta'

    if LOAD_FROM_PREV_SESS:
        print("Loading trained weights from previous session...")
        agent.init_infer(init_expert_weights)
    else:
        # initial training using expert policy
        print("Training from scratch, starting with expert...")
        use_expert_policy = True
        expert_data = {
            'observations': [],
            'actions': []
        }
        for i in range(num_expert_episodes):
            # TODO: for each changed param.....
            run_episode(env, agent, expert, expert_data, use_expert_policy)
        
        # train dagger using this initial expert data
        agent.train(expert_data)
        agent.save('init_expert')
        agent.init_infer(init_expert_weights)

    # now let agent execute commands to generate new set of data
    # from agent's own policy distribution
    use_expert_policy = False
    agent_data = {
        'observations': [],
        'actions': []
    }
    for i in range(num_agent_episodes):
        # TODO: for each changed param.....
        run_episode(env, agent, expert, agent_data, use_expert_policy)

    print(len(agent_data['observations']))
    agent.train(agent_data)
    agent.save('final_agent')
    agent.init_infer(final_agent_weights)

    # observe results
    print("FINAL RUN!!!")
    run_episode(env, agent, expert, agent_data, use_expert_policy)
    
    # Kp_vals = np.arange(start=5, stop=10, step=1.25)
    # Kd_vals = np.arange(start=0, stop=3, step=1.5)

    # parameters to vary:
    # spline aggressiveness
    # Kp and Kd values for PID
    # Q, R, S diagonals 
    # trajectories
    
    # first create a list of diffferent gateid series ~10
    # 7 train, 3 test
    # see if just one trajectory, perfect case works
    # next learn different trajectories
    # next modify spline aggressiveness

    # For OIL: modify Kp and Kd values enough to notice some difference

if __name__=='__main__':
    train_dagger()

