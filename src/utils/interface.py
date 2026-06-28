from abc import ABC as AbstractClass, abstractmethod

class Interface(AbstractClass):

    @abstractmethod
    def get_params(self):
        raise NotImplementedError(
            f"Must implement function get_params"
        )