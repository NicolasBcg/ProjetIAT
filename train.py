import sys
import time

from controller.dqn_agent import DQNAgent
from epsilon_profile import EpsilonProfile
from game.SpaceInvaders import SpaceInvaders

from networks import CNN

# test once by taking greedy actions based on Q values
def test_game(game: SpaceInvaders, agent: DQNAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = game.reset() if (same) else game.reset()


        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            state, reward, is_done = game.step(action)

            if display:
                time.sleep(speed)
                game.render()

            sum_rewards += reward
            if is_done:
                n_steps = step+1  # number of steps taken
                break
    return n_steps, sum_rewards


def main(nn, opt):
 
    """ INSTANCIE LE LABYRINTHE """ 
    game = SpaceInvaders(display=False)
    #  game = SpaceInvaders(display=True)
    
    #env.mode = "nn" 
    model = None
    gamma = 1.

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 40
    max_steps = 5000
    alpha = 0.001
    eps_profile = EpsilonProfile(1.0, 0.1)
    final_exploration_episode = 2000

    # Hyperparamètres de DQN
    batch_size = 32
    replay_memory_size = 2000
    target_update_frequency = 15
    tau = 1.0

    """ INSTANCIE LE RESEAU DE NEURONES """
    # Hyperparamètres 
    model = CNN(game.ny, game.nx, game.nf, game.na)#ny=450 maxy , nx=735 max x, nf=3 nombre de variables (joueur,bullet,invaders),na=4(nombre d'actions)
    

    print('--- neural network ---')
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('number of parameters:', num_params)
    print(model)

    """  LEARNING PARAMETERS"""
    agent = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    agent.learn(game, n_episodes, max_steps)
    game = SpaceInvaders(display=True)
    test_game(game, agent, max_steps=5000, nepisodes=10, speed=0.0001, display=True)

if __name__ == '__main__':
    """ Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """
    if (len(sys.argv) > 2):
        main(sys.argv[1], sys.argv[2:])
    if (len(sys.argv) > 1):
        main(sys.argv[1], [])
    else:
        main("random", [])