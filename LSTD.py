import pycolab
from pycolab import ascii_art
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab import human_ui
import random
import numpy as np
import pickle
from pycolab import things as plab_things
import matplotlib
from tabulate import tabulate
from tqdm import *
import time
from matplotlib import pyplot as plt
from IPython.display import clear_output
feature_vectors_bradke = np.array([
            [74.29, 34.61, 73.48, 53.29, 7.79],
            [61.60, 48.07, 34.68, 36.19, 82.02],
            [97.00, 4.88, 8.51, 87.89, 5.17],
            [41.10, 40.13, 64.63, 92.67, 31.09],
            [7.76, 79.82, 43.78, 8.56, 61.11]])

feature_vectors_boyan = np.array([
            [1, 0, 0, 0],
            [3/4, 1/4, 0, 0],
            [1/2, 1/2, 0, 0],
            [1/4, 3/4, 0, 0],
            [0, 1, 0, 0],
            [0, 3/4, 1/4, 0],
            [0, 1/2, 1/2, 0],
            [0, 1/4, 3/4, 0],
            [0, 0, 1, 0],
            [0, 0, 3/4, 1/4],
            [0, 0, 1/2, 1/2],
            [0, 0, 1/4, 3/4],
            [0, 0, 0, 0]])
feature_vectors_bradke = np.array([
            [74.29, 34.61, 73.48, 53.29, 7.79],
            [61.60, 48.07, 34.68, 36.19, 82.02],
            [97.00, 4.88, 8.51, 87.89, 5.17],
            [41.10, 40.13, 64.63, 92.67, 31.09],
            [7.76, 79.82, 43.78, 8.56, 61.11]])

feature_vectors_boyan = np.array([
            [1, 0, 0, 0],
            [3/4, 1/4, 0, 0],
            [1/2, 1/2, 0, 0],
            [1/4, 3/4, 0, 0],
            [0, 1, 0, 0],
            [0, 3/4, 1/4, 0],
            [0, 1/2, 1/2, 0],
            [0, 1/4, 3/4, 0],
            [0, 0, 1, 0],
            [0, 0, 3/4, 1/4],
            [0, 0, 1/2, 1/2],
            [0, 0, 1/4, 3/4],
            [0, 0, 0, 0]])


def make_game(environment, initial_state=0):
    """
    Initialize the MDP using Pycolab.
    """     
    if environment == 'bradke_chain':
        mdp_board = [" " * initial_state + "A" + " " * (4 - initial_state)]
        return ascii_art.ascii_art_to_game(art=mdp_board, 
                                       what_lies_beneath=' ',
                                       sprites={'A': Agent_bradke_chain})
    elif environment == 'boyan_chain':
        mdp_board = ['A           @']
        return ascii_art.ascii_art_to_game(art=mdp_board, 
                                       what_lies_beneath=' ',
                                       sprites={'A': Agent_boyan_chain})
    else:
        print('Wrong environment name!')
        

class Agent_bradke_chain(prefab_sprites.MazeWalker):
    """
    Agent for the 5 state MDP environment.
    """
    def __init__(self, corner, position, character):
        super(Agent_bradke_chain, self).__init__(corner, position, character, impassable=[])
        
        self.transition_matrix = np.array([
                           [0.42, 0.13, 0.14, 0.03, 0.28],
                           [0.25, 0.08, 0.16, 0.35, 0.15],
                           [0.08, 0.20, 0.33, 0.17, 0.22],
                           [0.36, 0.05, 0.00, 0.51, 0.07],
                           [0.17, 0.24, 0.19, 0.18, 0.22]])
        
        self.reward_matrix = np.array([
                            [104.66, 29.69, 82.36, 37.49, 68.82],
                            [75.86, 29.24, 100.37, 0.31, 35.99],
                            [57.68, 65.66, 56.95, 100.44, 47.63],
                            [96.53, 14.01, 0.88, 89.77, 66.77],
                            [70.35, 23.69, 73.41, 70.70, 85.41]]) 
            

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things  
        
        if actions == 0: # Avoid acting when game.its_showtime() is called
            # Go to a state given by the transition matrix
            current_state = self.position.col
            next_state = np.argmax(np.random.multinomial(1, self.transition_matrix[current_state]))
            self._teleport((0, next_state))

            # Reward from reward matrix
            the_plot.add_reward(self.reward_matrix[current_state, next_state])
 

class Agent_boyan_chain(prefab_sprites.MazeWalker):
    """
    Agent for the 13-state Boyan's chain environment.
    """
    def __init__(self, corner, position, character):
        super(Agent_boyan_chain, self).__init__(corner, position, character, impassable=[])
           
    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things  

        if actions == 0: # Avoid acting when game.its_showtime() is called
            # Transition to goal if the agent is in the state adjacent to the goal
            if layers["@"][0, self.position.col + 1]:
                the_plot.add_reward(-2)
                self._east(board, the_plot)

            # The makes 1 jump east with prob. 0.5 and 2 jumps east with 0.5 prob.
            else:
                the_plot.add_reward(-3.0)
                if np.random.rand() >= 0.5:
                    self._east(board, the_plot)
                else:
                    self._east(board, the_plot)
                    self._east(board, the_plot)

            # Terminate episode if the agent reaches the goal
            if layers["@"][0, self.position.col]:
                the_plot.terminate_episode()

            
def show_board(observation, fig, ax, environment):
    '''
    Display the board. Uses ion to display in a Jupyter Notebook.
    Adapted from previous assignment.
    '''
    plt.ion()    
    fig.show()
    fig.canvas.draw()
    
    # Paint board
    grid = np.array(observation.layers[' '], dtype=np.float)
    grid += 60 * np.array(observation.layers['A'], dtype=np.float)
    
    if environment == "boyan_chain":
        # Goal
        grid += 90 * np.array(observation.layers['@'], dtype=np.float)
    
    ax.clear()
    ax.axis('off')
    ax.imshow(grid, cmap='viridis')
    fig.canvas.draw()
    time.sleep(0.2)
def featurizer(state, features):
    """
    Return the featurized representation of a state.
    """
    return np.expand_dims(features[state], 1)


def get_state(observation):
        """
        Returns the current state given a Pycolab observation.
        """
        return np.argwhere(observation.layers['A'][0])[0][0]


def initialize_episode(environment, features, initial_state):
        """
        Initialize the Pycolab engine given an initial state.
        """            
        if initial_state is None and environment == 'bradke_chain':
                current_state = np.random.randint(0, 5)
        elif environment == 'bradke_chain':
            current_state = initial_state
                
        # Boyan's chain initial state is always 0
        else: #environment == 'boyan_chain'
            current_state = 0

        # Instantiate game
        game = make_game(initial_state=current_state, environment=environment)
        observation, reward, discount = game.its_showtime()

        current_state_feature = featurizer(current_state, features)
        z = current_state_feature
        return game, observation, reward, discount, current_state_feature, z
    
    
class LSTD(object):
    """
    Contains methods to evaluate a policy using LSTD(\lambda) offline or incrementally. 
    Arguments:
        features (np.array) -- |S| x |K| matrix that contains the feature mappings of the states.
        environment (str) -- "bradke_chain" or "boyan_chain". 
    """
    def __init__(self, environment='bradke_chain', store_per_episode=False):
        self.environment = environment
        if self.environment == 'bradke_chain':
            self.features = feature_vectors_bradke
        elif self.environment == 'boyan_chain':
            self.features = feature_vectors_boyan
        else:
            print('Wrong environment name!')
        self.num_features = self.features.shape[1]
        self.num_states = self.features.shape[0]    
        self.store_per_episode = store_per_episode
        
    
    def offline(self, lambda_, gamma, max_timesteps, num_episodes,
                initial_state=None, return_A_b=False, render=False):
        """
        Evaluate the policy during num_episodes using offline LSTD(\lambda).
        """     
        A = np.zeros((self.num_features, self.num_features))
        b = np.zeros((self.num_features, 1))
        
        if self.store_per_episode:
            self.theta_per_episode = np.zeros((num_episodes, self.num_features)) 

        #for episode in tqdm_notebook(range(self.num_episodes)):
        for episode in range(num_episodes):
            t = 0        
            game, observation, reward, discount, current_state_feature, z = initialize_episode(self.environment, 
                                                                         self.features, initial_state)
            if render:
                fig, ax = plt.subplots()
                show_board(observation, fig, ax, environment=self.environment)
                
            while not game.game_over and t < max_timesteps:
                # Step 
                t += 1
                observation, reward, discount = game.play(0)
                next_state = get_state(observation)
                next_state_feature = featurizer(next_state, self.features)

                A += np.outer(z, current_state_feature - gamma * next_state_feature)
                b += z * reward
                z = lambda_ * z + next_state_feature

                # x_t <-- x_{t+1} and \phi(x_t) <-- \phi(x_{t+1})
                current_state = next_state
                current_state_feature = featurizer(current_state, self.features)

                if render:
                    show_board(observation, fig, ax, environment=self.environment)

            # Estimate values at the end of episode     
            self.theta = np.dot(np.linalg.inv(A), b)
            self.V = np.dot(self.features, self.theta)
          
            if self.store_per_episode:
                self.theta_per_episode[episode] = self.theta.T

        if return_A_b:
            return A, b,
           
            
    def incremental(self, lambda_, gamma, max_timesteps, num_episodes, 
                    epsilon=0.0001, initial_state=None, return_A_b=False, render=False):
        """
        Evaluate the policy during num_episodes using incremental LSTD(\lambda).
        """  
        A = np.zeros((self.num_features, self.num_features))
        A_inverse = 1 / epsilon * np.identity(self.num_features)
        b = np.zeros((self.num_features, 1))
        
        if self.store_per_episode:
            theta_per_episode = np.zeros((num_episodes, self.num_features)) 
            
        #for episode in tqdm_notebook(range(self.num_episodes)):
        for episode in range(num_episodes):
            t = 0       
            game, observation, reward, discount, current_state_feature, z = initialize_episode(self.environment, 
                                                                         self.features, initial_state)  
            if render:
                fig, ax = plt.subplots()
                show_board(observation, fig, ax, environment=self.environment)
            
            while not game.game_over and t < max_timesteps:
                # Step
                t += 1
                observation, reward, discount = game.play(0)
                next_state = get_state(observation)
                next_state_feature = featurizer(next_state, self.features)

                b += z * reward
                v = np.dot(A_inverse.T, current_state_feature - gamma * next_state_feature)
                A_inverse -= (np.dot(np.dot(A_inverse, z), v.T)) / (1 + np.dot(v.T, z))
                z = lambda_ * z + next_state_feature

                # x_t <-- x_{t+1} and \phi(x_t) <-- \phi(x_{t+1})
                current_state = next_state
                current_state_feature = featurizer(current_state, self.features)

                if render:
                    show_board(observation, fig, ax, environment=self.environment)

            # Estimate values at the end of episode
            self.theta = np.dot(A_inverse, b)
            self.V = np.dot(self.features, self.theta)
          
            if self.store_per_episode:
                self.theta_per_episode[episode] = self.theta.T
            
        if return_A_b:
            return np.linalg.pinv(A_inverse), b
class TD(object):
    """
    Implementation of TD(lambda).
    """
    def __init__(self, environment, store_per_episode=False):
        self.environment = environment        
        if self.environment == 'bradke_chain':
            self.features = feature_vectors_bradke
        elif self.environment == 'boyan_chain':
            self.features = feature_vectors_boyan
        self.num_features = self.features.shape[1]
        self.store_per_episode = store_per_episode
    
    
    def run(self, lambda_, gamma, a0, n0, max_timesteps, num_episodes):
        self.theta = np.zeros((self.num_features, 1))

        if self.store_per_episode:
            self.theta_per_episode = np.zeros((num_episodes, self.num_features)) 
            
        #t = 0
        for episode in range(num_episodes):
            t_eps = 0      
            game, observation, reward, discount, current_state_feature, z = initialize_episode(self.environment, 
                                                                         self.features, 0)
            z = current_state_feature
            d = np.zeros((self.num_features, 1))
            
            while not game.game_over and t_eps < max_timesteps:
                # Step 
                t_eps += 1
                #t += 1
                alpha = a0 * (n0 + 1) / (n0 + episode+1)

                observation, reward, discount = game.play(0)
                next_state = get_state(observation)
                next_state_feature = featurizer(next_state, self.features)
                
                d += z * (reward + ((gamma*next_state_feature - current_state_feature).T.dot(self.theta)))
                z = gamma*lambda_*z + next_state_feature
                          
                current_state = next_state
                current_state_feature = featurizer(current_state, self.features)
                
                self.theta += alpha * d
                
                if self.store_per_episode:
                    self.theta_per_episode[episode] = self.theta.T
            

def solve_bradke(gamma):
    """
    Solve the linear system of equations for \theta^* in Bradke's chain.
    """
    # Environment dynamics
    transition_matrix = np.array([
                               [0.42, 0.13, 0.14, 0.03, 0.28],
                               [0.25, 0.08, 0.16, 0.35, 0.15],
                               [0.08, 0.20, 0.33, 0.17, 0.22],
                               [0.36, 0.05, 0.00, 0.51, 0.07],
                               [0.17, 0.24, 0.19, 0.18, 0.22]])

    reward_matrix = np.array([
                            [104.66, 29.69, 82.36, 37.49, 68.82],
                            [75.86, 29.24, 100.37, 0.31, 35.99],
                            [57.68, 65.66, 56.95, 100.44, 47.63],
                            [96.53, 14.01, 0.88, 89.77, 66.77],
                            [70.35, 23.69, 73.41, 70.70, 85.41]]) 

    r_bar = np.sum(reward_matrix * transition_matrix, axis=1)

    true_theta_bradke= np.linalg.inv(feature_vectors_bradke) \
                .dot(np.linalg.inv(np.identity(5) - gamma * transition_matrix)) \
                .dot(r_bar)
    true_theta_bradke = np.expand_dims(true_theta_bradke, 1)   
    return true_theta_bradke


def plot_fixed_gamma(environment, thetas, implementation, gamma_list=[0.3, 0.6, 0.9, 1]):
    """
    Auxiliary function to plot experiments.
    """           
    # Plot
    for gamma in gamma_list:
        fig, ax = plt.subplots(figsize=(12,5))

        for key in thetas[environment]:
            if environment == 'boyan_chain':
                true_theta = np.array([-24, -16, -8, 0]).reshape(4, 1)
                feature_vectors = feature_vectors_boyan

            elif environment == 'bradke_chain':
                true_theta = solve_bradke(gamma)
                feature_vectors = feature_vectors_bradke
                
            # Fix gamma and vary lambda
            if key[1] == gamma:
                #ax.semilogx(rmse(thetas[environment][key], true_theta.T),
                #            '.-', label="$\lambda = {}$".format(key[0]))
                
                # Compute RMSE w.r.t. V^*
                ax.semilogx(rmse(thetas[environment][key].dot(feature_vectors.T), 
                                 feature_vectors.dot(true_theta).T), 
                            '.-', label="$\lambda = {}$".format(key[0]))

        ax.set_title('LSTD($\lambda$): {} implementation, $\gamma={}$'
                     .format(implementation, gamma), fontsize=13)
        ax.set_xlabel('Trajectory number', fontsize=13)
        ax.set_ylabel('Average RMSE of $V$ over 10 trials', fontsize=13)
        ax.legend()
        plt.grid()
    plt.show()
  

def run_experiments(lambdas, gammas, num_episodes=10000, num_trials=10, max_timesteps=100, epsilon=0.0001, 
                    initial_state=0, environments=['bradke_chain', 'boyan_chain']):
    """
    Arguments:
        lambdas (list) -- list of bias/variance parameters to try.
        gammas (list) -- list of discount factors to try.
    """
    # Store results in dictionnaries: {env: {(lambda, gamma): theta, (lambda, gamma): theta, ...}, env: ...}
    results_offline = dict()
    results_incremental = dict()
        
    for env in tqdm_notebook(environments):
        results_offline[env] = dict()
        results_incremental[env] = dict()
        
        lstd = LSTD(env, store_per_episode=True)
        
        # Perform only one "episode" in Bradke chain
        if env == "bradke_chain":
            n_episodes = 1
        else:
            n_episodes = num_episodes
        
        for lambda_ in lambdas:
            for gamma in gammas:
                results_offline[env][(lambda_, gamma)] = np.zeros((num_episodes, lstd.num_features))
                results_incremental[env][(lambda_, gamma)] = np.zeros((num_episodes, lstd.num_features))

                # Average over trials
                for trial in range(num_trials):
                    
                    # Offline LSTD(\lambda)
                    lstd.offline(lambda_, gamma, max_timesteps, 
                                 num_episodes=num_episodes, initial_state=initial_state)
                    results_offline[env][(lambda_, gamma)] += lstd.theta_per_episode

                    # Incremental LSTD(\lambda)
                    lstd.incremental(lambda_, gamma, max_timesteps, 
                                     epsilon=epsilon, num_episodes=num_episodes, 
                                     initial_state=initial_state)
                    results_incremental[env][(lambda_, gamma)] += lstd.theta_per_episode
                    
                results_offline[env][(lambda_, gamma)] /= num_trials
                results_incremental[env][(lambda_, gamma)] /= num_trials
                
    return results_offline, results_incremental


def plot_td_experiment(results_lstd, thetas, lambda_=0.4):
    """
    Auxiliary function to plot experiments.
    """           
    # Plot
    fig, ax = plt.subplots(figsize=(12,5))
    markers = ['x', '^', '+', 'd', 's', '*']
    step = 1
    x_values = np.arange(10000)[::step]
    
    for i, key in enumerate(thetas):
        true_theta = np.array([-24, -16, -8, 0]).reshape(4, 1)
        feature_vectors = feature_vectors_boyan

        # Compute RMSE w.r.t. V^*
        ax.semilogx(x_values, rmse(thetas[key][::step].dot(feature_vectors.T), feature_vectors.dot(true_theta).T), 
                    '.-', label="TD: $a_0 = {}, n_0 = {}$".format(key[0], key[1]), 
                    alpha=0.6, marker=markers[i])
    
    ax.semilogx(x_values, rmse(results_lstd[0]['boyan_chain'][(0.4, 1)][::step].dot(feature_vectors.T),
                     feature_vectors.dot(true_theta).T), label='LSTD')
    
    ax.set_xlabel('Trajectory number', fontsize=13)
    ax.set_ylabel('Average RMSE of $V$ over 10 trials', fontsize=13)
    ax.set_ylim([0, 2])
    ax.set_xlim(xmax=10000)
    ax.set_title('$\lambda = {}$ $\gamma={}$'
                 .format(0.4, 1), fontsize=13)
    ax.legend()
    plt.grid()
    plt.show()
    

def td_experiment(lambda_=0.4, gamma=1, num_episodes=10000, n_runs = 10):
    # Store thetas from TD. The keys are (a0, n0)
    results_td = {(0.1, 10**6): np.zeros((num_episodes, 4)), (0.1, 10**3): np.zeros((num_episodes, 4)),
                  (0.1, 10**2): np.zeros((num_episodes, 4)), (0.01, 10**6): np.zeros((num_episodes, 4)),
                  (0.01, 10**3): np.zeros((num_episodes, 4)), (0.01, 10**2): np.zeros((num_episodes, 4))}

    td = TD('boyan_chain', store_per_episode=True)

    for key in tqdm_notebook(results_td.keys()):
        for run in range(n_runs):
            td.run(lambda_=lambda_, gamma=gamma, a0=key[0], n0=key[1], max_timesteps=100, num_episodes=10000)
            results_td[key] += td.theta_per_episode
        results_td[key] /= n_runs

    # Save dictionnary
    save_obj(results_td, 'results_td')
    return results_td


def rmse(a, b):
    """
    Compute the root mean squared error between a and b.
    """
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



def linear_regression(environment, gamma, num_episodes, max_timesteps):
    """
    Arguments:
        environment (str) -- "bradke_chain" or "boyan_chain".
        gamma (float) -- discount factor.
        num_episodes (int) -- number of evaluation episodes.
        max_timesteps (int) -- maximum number of steps in an episode.        
    """
    if environment == 'bradke_chain':
        features = feature_vectors_bradke
        num_features = 5
    elif environment == 'boyan_chain':
        features = feature_vectors_boyan
        num_features = 4
    else:
        print('Wrong environment name. Must be "bradke_chain" or "boyan_chain".')
        
    A = np.zeros((num_features, num_features))
    b = np.zeros((num_features, 1))
    
    for episode in range(num_episodes):
        # Instantiate game
        game, observation, reward, discount, current_state_feature, z = initialize_episode(environment, features, 0)
        t = 0
        
        feature_history = current_state_feature.T   # Store \phi(x_0), \phi(x_1), ..., \phi(x_L) in each row
        reward_history = []                         # Store (R_0+R_1+...+R_L), (R_1+...+R_L), ..., R_L
        gamma_history = []
        
        A += np.outer(current_state_feature, current_state_feature)  # Initial state
        
        # Run one episode 
        while not game.game_over and t < max_timesteps:            
            # Step
            t += 1
            observation, reward, discount = game.play(0)
            next_state = get_state(observation)
            next_state_feature = featurizer(next_state, features)

            A += np.outer(next_state_feature, next_state_feature)

            # Store \phi(x_t) and R_t
            feature_history = np.vstack((feature_history, next_state_feature.T))
            reward_history.append(reward)

        # Compute b
        for i in range(t):
            y_i = np.power(gamma, range(t - i)).dot(reward_history[i:])
            b += feature_history[i, None].T * y_i

    return A, b, np.linalg.inv(A).dot(b) # theta





lambdas = [0, 0.2, 0.4, 0.6, 0.8, 1]
gammas = [0.3, 0.6, 0.9, 1]

# Beware: this takes a while to run!
if 1 == True: # avoid running by accident
    offline_results, incremental_results = run_experiments(lambdas, gammas,
                                                           num_episodes=10000, num_trials=10,
                                                           max_timesteps=50, initial_state=0,
                                                           environments=['boyan_chain', 'bradke_chain'])




plot_fixed_gamma('boyan_chain', offline_results, 'Offline', gamma_list=[1])
#plot_fixed_gamma('bradke_chain', incremental_results, 'Incremental', gamma_list=[0.3, 0.6, 0.9])

seed = 123
# Dictionary to store the A, b and thetas
quantities = {'boyan_chain': {'offline': {'A': 0, 'b':0, 'theta':0}, 
                              'incremental': {'A':0, 'b':0, 'theta':0}, 
                              'lr': {'A':0, 'b':0, 'theta':0}},  
              'bradke_chain': {'offline': {'A': 0, 'b':0, 'theta':0}, 
                               'incremental': {'A':0, 'b':0, 'theta':0}, 
                               'lr': {'A':0, 'b':0, 'theta':0}}}

############################Starting to output parameters and comparison of A, b, ....

## Boyan chain
## LSTD(1)
# lstd = LSTD('boyan_chain')
# np.random.seed(seed)
# A, b = lstd.offline(lambda_=1, gamma=1, max_timesteps=100, 
#                                           num_episodes=1000, initial_state=0, return_A_b=True)
# quantities['boyan_chain']['offline']['A'] = A
# quantities['boyan_chain']['offline']['b'] = b
# quantities['boyan_chain']['offline']['theta'] = lstd.theta
# quantities['boyan_chain']['offline']['V'] = lstd.V

# np.random.seed(seed)
# A, b = lstd.incremental(lambda_=1, gamma=1, max_timesteps=100, 
#                                           num_episodes=1000, initial_state=0, return_A_b=True)
# quantities['boyan_chain']['incremental']['A'] = A
# quantities['boyan_chain']['incremental']['b'] = b
# quantities['boyan_chain']['incremental']['theta'] = lstd.theta
# quantities['boyan_chain']['incremental']['V'] = lstd.V

# # Linear regression
# np.random.seed(seed)
# A, b, theta = linear_regression('boyan_chain', gamma=1, num_episodes=1000, max_timesteps=100)
# quantities['boyan_chain']['lr']['A'] = A
# quantities['boyan_chain']['lr']['b'] = b
# quantities['boyan_chain']['lr']['theta'] = theta
# quantities['boyan_chain']['lr']['V'] = np.dot(feature_vectors_boyan, theta)

# #############

# # Bradke chain
# # LSTD(1)
# lstd = LSTD('bradke_chain')
# np.random.seed(seed)
# A, b = lstd.offline(lambda_=1, gamma=0.5, max_timesteps=100, 
#                                           num_episodes=1000, initial_state=0, return_A_b=True)
# quantities['bradke_chain']['offline']['A'] = A
# quantities['bradke_chain']['offline']['b'] = b
# quantities['bradke_chain']['offline']['theta'] = lstd.theta
# quantities['bradke_chain']['offline']['V'] = lstd.V

# np.random.seed(seed)
# A, b = lstd.incremental(lambda_=1, gamma=0.5, max_timesteps=100, 
#                                           num_episodes=1000, initial_state=0, return_A_b=True)
# quantities['bradke_chain']['incremental']['A'] = A
# quantities['bradke_chain']['incremental']['b'] = b
# quantities['bradke_chain']['incremental']['theta'] = lstd.theta
# quantities['bradke_chain']['incremental']['V'] = lstd.V

# # Linear regression
# np.random.seed(seed)
# A, b, theta = linear_regression('bradke_chain', gamma=0.5, num_episodes=100, max_timesteps=100)
# quantities['bradke_chain']['lr']['A'] = A
# quantities['bradke_chain']['lr']['b'] = b
# quantities['bradke_chain']['lr']['theta'] = theta
# quantities['bradke_chain']['lr']['V'] = np.dot(feature_vectors_bradke, theta)

