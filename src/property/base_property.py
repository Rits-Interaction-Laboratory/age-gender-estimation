import yaml


class BaseProperty:
    """
    プロパティの基底クラス
    """

    def __init__(self):
        base_property_name: str = type(self).__name__.replace("Property", "").lower()
        
        with open("resources/application.yml", "r") as file:
            properties = yaml.safe_load(file)
            for key, value in properties[base_property_name].items():
                if isinstance(value, str):
                    exec(f"self.{key} = \"{value}\"")
                else:
                    exec(f"self.{key} = {value}")
