from gymnasium import register


register(
    id='pusht-v0',
    entry_point='relax.env.pusht.pusht_env:PushTEnv',
    max_episode_steps=300
)

register(
    id='pusht-shp-v0',
    entry_point='relax.env.pusht.pusht_shp_env:PushTShpEnv',
    max_episode_steps=300
)
