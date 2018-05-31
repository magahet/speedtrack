'''Provides a config class.'''

import yaml


class Config(object):
    """Represents the app config."""

    def __init__(self, config_path):
        self.config = None
        self.load_config(config_path)

    def __getattr__(self, attr):
        attr = attr.lower()
        if attr not in self.config:
            raise AttributeError(
                '[{}] is missing from the config file'.format(attr))
        return self.config.get(attr)

    def load_config(self, config_path):
        """Load config from yaml file."""
        with open(config_path) as config_file:
            self.config = yaml.safe_load(config_file)

    def get(self, key):
        """Get config value."""
        return self.config.get(key)

    def update(self, key, value):
        """Update config value."""
        self.config[key] = value
