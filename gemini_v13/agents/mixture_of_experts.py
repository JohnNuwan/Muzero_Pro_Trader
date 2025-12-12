from collections import Counter
from .base_agent import BaseAgent

class MixtureOfExpertsAgent(BaseAgent):
    """
    Le 'Conseil des Sages'.
    Cet agent ne décide pas seul. Il consulte une liste d'experts (sous-agents)
    et prend la décision finale par vote majoritaire.
    """
    def __init__(self, experts, name="Council_of_Elders"):
        super().__init__(name=name)
        self.experts = experts # Liste d'instances de QLearningAgent
        self.last_vote_details = ""

    def act(self, obs):
        """
        Demande à chaque expert son avis et vote.
        """
        votes = []
        for expert in self.experts:
            action = expert.act(obs)
            votes.append(action)
            
        # Vote Majoritaire
        vote_counts = Counter(votes)
        final_action = vote_counts.most_common(1)[0][0]
        
        # Détails pour le log (ex: "3xBuy, 2xHold")
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
        details = []
        for action, count in vote_counts.items():
            details.append(f"{count}x{action_names.get(action, str(action))}")
        self.last_vote_details = ", ".join(details)
        
        return final_action

    def learn(self, obs, action, reward, next_obs, done):
        """
        Le MoE n'apprend pas directement (pour l'instant).
        Il est composé d'experts déjà entraînés (figés).
        On pourrait imaginer faire apprendre les experts individuellement,
        mais pour l'instant on utilise les modèles 'Champions' figés.
        """
        pass
