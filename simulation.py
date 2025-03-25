from main import Environment, Player1, Player2, Game

env = Environment(
    size=(6, 6),
    mode="debug"
)

p1 = Player1(
    env=env
)

p2 = Player2(
    env=env
)

game = Game(
    env=env,
    player1=p1,
    player2=p2,
    B=0.5,
    C=0.5
)

env.assign_obstacles()
env.assign_players()
env.save_to_json()

game.run(num_episodes=1000, T=10)
game.export_trajectory(filename="test_debug_simulation.json")
