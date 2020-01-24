from gym.envs.registration import register

register(
    id='Pushing2D-v0',
    entry_point='envs.2Dpusher_env:Pusher2d'
)

register(
    id='Point2D-v0',
    entry_point='envs.2d_point:PointEnv'
)
