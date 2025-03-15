from .learning_parameters import *
from .training_utils import *


import numpy as np

import torch.nn as nn
import torch

def epsilon(steps):
    if steps<EPS_OFFSET:
        return EPS_START
    else:
        return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * (steps-EPS_OFFSET) / EPS_DECAY)

def main(environement, 
         device, 
         model_actor:nn.Module, 
         model_critic:nn.Module, 
         num_episodes:int = 10000, 
         num_item_in_memory:int = 20000):
    
    actor = model_actor()
    critic = model_critic()
    
    critic_smooth = model_critic()
    soft_update(net=critic_smooth, target_net=critic, tau=1.0)
    
 
    memory = ReplayMemory(num_item_in_memory)
    memory_test = ReplayMemory(num_item_in_memory)
    proba_test_set = .2 #20% des observations d'une game vont dans le test set. 

    steps_done = 0
    score_hero = []
    
    l_loss_actor = [0]
    l_loss_critic = [0]
    l_loss_actor_test = [0]
    l_loss_critic_test = [0]

    l_epsilon = []

    regenerate_dataset=0
    for i_episode in range(num_episodes):

        state = environement.reset()

        
        while True :
            steps_done += 1
            action = select_action(state, actor, device, epsilon(steps_done))

            next_state,_, reward, done = env.step(state=state, action=action)
            action_hero = action_hero.unsqueeze(0)
            action_adv = action_adv.unsqueeze(0)
            
            next_state_hero = torch.tensor(observation_hero, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_adversaire = torch.tensor(observation_adversaire, dtype=torch.float32, device=device).unsqueeze(0)

            if random.rand()<proba_test_set:
                memory_test.push(state_hero, action_hero, action_adv, next_state_hero, state_hero[:, -1])
            else:
                memory.push(state_hero, action_hero, action_adv, next_state_hero, state_hero[:, -1])
            # memory.push(state_adversaire, action_adv, action_hero, next_state_adversaire, next_state_adversaire[:, -1])

            predictided_future_reward = hero.critic.net(next_state_hero).clone().detach()
            future_reward = hero.critic.net(torch.cat((state_hero[:, :20], predictor_net(state_hero, action_hero, action_adv)), dim=1)).clone().detach()

            critic_sum += predictided_future_reward
            critic_expected_sum += future_reward
            eqm_critic_sum += (predictided_future_reward-future_reward)**2
            hard_critic_sum += next_state_hero[0, -1]
            
            state_hero = next_state_hero
            state_adversaire = next_state_adversaire

            if regenerate_dataset==0 and random.rand()<rate_training:
                train_models(memory=memory, 
                             model_training=model_training, 
                             loss_pred_dq=loss_pred_dq, 
                             loss_critic_dq=loss_critic_dq, 
                             loss_actor_dq=loss_actor_dq, 
                             steps_done=steps_done, 
                             last_loss_predictor=average(loss_pred_dq),
                             last_loss_critic=average(loss_critic_dq), 
                             test_dataset=False)
                train_models(memory=memory_test, 
                             model_training=model_training, 
                             loss_pred_dq=loss_pred_test_dq, 
                             loss_critic_dq=loss_critic_test_dq, 
                             loss_actor_dq=loss_actor_test_dq, 
                             steps_done=steps_done, 
                             last_loss_predictor=average(loss_pred_dq),
                             last_loss_critic=average(loss_critic_dq), 
                             test_dataset=True)
            soft_update(hero.critic.net, hero.critic_smooth.net)
            

            if terminated:  
                if eqm_critic_sum>1000:
                    eqm_critic_sum=1000
            

                if regenerate_dataset>0: 
                    regenerate_dataset-=1
                else:
                    update_losses_values(l_loss_actor, l_loss_critic, l_loss_predictor, loss_actor_dq, loss_critic_dq, loss_pred_dq)
                    update_losses_values(l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test, loss_actor_test_dq, loss_critic_test_dq, loss_pred_test_dq)
                    
                l_gamma.append(gamma(steps_done))
                l_epsilon.append(epsilon(steps_done))

                n_cp_hero, n_cp_adversaire = env.get_cp()

                score_hero.append(n_cp_hero)
                score_adv.append(n_cp_adversaire)

                critic_value.append(float(critic_sum)/t)
                critic_value_expected.append(float(critic_expected_sum)/t)
                eqm_critic_liste.append(float(eqm_critic_sum)/t)
                hard_critic_value.append(float(hard_critic_sum)/t)
                
                
                torch.save(hero.actor.net.state_dict(), f'./models/actor/{hero.actor.version}/safetensor/{hero.actor.save_name}')
                torch.save(hero.critic.net.state_dict(), f'./models/critic/{hero.critic.version}/safetensor/{hero.critic.save_name}')
                torch.save(predictor_net.state_dict(), f'./models/predictor/{"version1"}/safetensor/gen_1_working')

                plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor, l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test)

                break


    print('Complete')
    # plot_infos(score_hero, score_adv, l_loss_actor, l_loss_critic, l_gamma, l_epsilon, eqm_critic_liste, critic_value, critic_value_expected, hard_critic_value, l_loss_predictor, l_max_loss_predictor, l_loss_actor_test, l_loss_critic_test, l_loss_predictor_test)
    plt.ioff()
    plt.show()
    print(score_hero)

if __name__ == "__main__":
    main()