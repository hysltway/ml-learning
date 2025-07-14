import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.train_eps =  5000      # 训练的回合数
        self.policy_lr = 0.05      # 学习率
        self.gamma = 0.9          # 折扣因子
        self.epsilon = 0.1        # ε-贪婪策略中的ε

cfg=Config()

class CliffWalking:
    def __init__(self,env):
        self.env=env
        # 这两个参数虽然后面没有直接用到，但在初始化 QLearning 时需要用到 observation_space.n 和 action_space.n
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self):
        # 处理新版gymnasium的reset返回值(observation, info)
        state, info = self.env.reset()
        return state

    def step(self,action):
        # 处理新版gymnasium的step返回值(next_state, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info

class QLearning:
    def __init__(self,state,action,learning_rate,gamma,epsilon):
        self.state=state
        self.action=action
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.epsilon=epsilon
        self.Q=np.zeros((state,action))

    def take_action(self,state):
        # np.random.random() 的取值范围是 [0.0, 1.0)
        if np.random.random() < self.epsilon:
            action=np.random.randint(self.action)
        else:
            # np.argmax返回Q[state]中最大值的索引，即选择Q值最大的动作
            action = np.argmax(self.Q[state])
        return action
    
    def update(self,state,action,reward,next_state):
        # Q-learning更新公式: Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]
        # 1. 选择下一个状态的最大Q值对应的动作
        next_action = np.argmax(self.Q[next_state])
        # 2. 计算目标Q值：即时奖励 + 折扣因子 * 下一个状态的最大Q值
        td_target = reward + self.gamma * self.Q[next_state, next_action]
        # 3. 计算TD误差：目标Q值 - 当前Q值
        td_error = td_target - self.Q[state, action]
        # 4. 按照学习率对当前Q值进行更新
        self.Q[state, action] += self.learning_rate * td_error

env = gym.make('CliffWalking-v0')
env=CliffWalking(env)


agent=QLearning(state=env.observation_space.n,
                action=env.action_space.n,
                learning_rate=cfg.policy_lr,
                gamma=cfg.gamma,
                epsilon=cfg.epsilon)

rewards = []  # 改名为rewards避免与单步奖励冲突
ma_reward=[]


for i in range(cfg.train_eps):
    ep_reward=0
    state=env.reset()
    while True:
        action=agent.take_action(state)
        next_state, r, done, info=env.step(action)  # 使用r代替reward作为单步奖励
        agent.update(state, action, r, next_state)
        state=next_state
        ep_reward+=r
        if done:
            break
    rewards.append(ep_reward)  # 使用新命名的rewards列表
    if ma_reward:
        ma_reward.append(0.9*ma_reward[-1]+0.1*ep_reward)
    else:
        ma_reward.append(ep_reward)
    print(f"Episode: {i}, Reward: {ep_reward}, Moving Average Reward: {ma_reward[-1]}")
    
plt.plot(rewards)  # 使用新命名的rewards列表
plt.plot(ma_reward)
plt.title("Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

