import toml

with open("src/example_config.toml", "r") as f:
    config = toml.load(f)

print(config)
