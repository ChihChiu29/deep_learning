"""For defining policy."""


class PolicyInterface():
    
    def MakeDecision(self, situation):
        """Makes a decision.
        
        Args:
            situation: same as the "observation" from gym.
            
        Returns:
            A decision that belongs to `env.action_space`.
        """
        raise NotImplementedError()
        
    
class ValueFunctionInterface():
    
    def ValueFor(self, situation, action) -> float:
        """Gives the value for a situation and an action.
        
        Args:
            situation: same as the "observation" from gym.
            action: an action that belongs to `env.action_space`.
            
        Returns:
            A float for the currently stored value.
        """
        raise NotImplementedError()