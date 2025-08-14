"""
D&D Session Simulation Template
Simulates tabletop RPG sessions with focus on:
- Creative problem-solving and storytelling
- Player engagement and interaction
- Narrative coherence and world-building
- Rule interpretation and adaptation
"""

from core.interfaces.agent_interface import AgentContext, AgentCapability
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random

class DnDRole(Enum):
    DUNGEON_MASTER = "dungeon_master"
    PLAYER_FIGHTER = "player_fighter"
    PLAYER_WIZARD = "player_wizard"
    PLAYER_ROGUE = "player_rogue"
    PLAYER_CLERIC = "player_cleric"
    NPC_VILLAGER = "npc_villager"
    NPC_MERCHANT = "npc_merchant"
    NPC_VILLAIN = "npc_villain"

@dataclass
class DnDCharacter:
    """D&D character sheet and traits"""
    name: str
    character_class: str
    level: int
    stats: Dict[str, int]  # STR, DEX, CON, INT, WIS, CHA
    skills: List[str]
    equipment: List[str]
    backstory: str
    personality_traits: List[str]
    goals: List[str]
    relationships: Dict[str, str]

class DnDSimulation:
    """D&D session environment simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_state = {
            "current_scene": "tavern_meeting",
            "location": "The Prancing Pony Tavern",
            "time_of_day": "evening",
            "weather": "light_rain",
            "party_status": {
                "health": "full",
                "resources": "well_stocked",
                "morale": "high",
                "cohesion": 8  # 1-10 scale
            },
            "story_progress": {
                "main_quest": "find_missing_artifact",
                "side_quests": ["help_merchant", "investigate_rumors"],
                "completed_objectives": [],
                "current_challenge": "gathering_information"
            },
            "world_events": [],
            "npc_reactions": {},
            "dice_rolls": []
        }
        
        self.campaign_world = {
            "setting": "fantasy_medieval",
            "locations": {
                "tavern": {"npcs": ["bartender", "merchant", "mysterious_stranger"], "atmosphere": "cozy"},
                "forest": {"dangers": ["bandits", "wild_animals"], "resources": ["herbs", "game"]},
                "dungeon": {"challenges": ["traps", "monsters", "puzzles"], "rewards": ["treasure", "artifacts"]},
                "town": {"services": ["shops", "temple", "inn"], "politics": "neutral"}
            },
            "lore": {
                "history": "ancient_empire_ruins",
                "conflicts": "war_between_kingdoms",
                "mysteries": "disappeared_civilization"
            }
        }
        
        self.available_tools = [
            "dice_roller",
            "rule_lookup",
            "character_sheet",
            "inventory_management",
            "spell_casting",
            "skill_check",
            "combat_system",
            "narrative_generator",
            "npc_generator",
            "world_builder"
        ]
    
    def get_scenario_context(self, agent_role: DnDRole) -> AgentContext:
        """Generate context specific to D&D role"""
        
        # Role-specific tool access
        role_tools = {
            DnDRole.DUNGEON_MASTER: [
                "dice_roller", "rule_lookup", "narrative_generator", 
                "npc_generator", "world_builder", "combat_system"
            ],
            DnDRole.PLAYER_FIGHTER: [
                "dice_roller", "character_sheet", "inventory_management", 
                "skill_check", "combat_system"
            ],
            DnDRole.PLAYER_WIZARD: [
                "dice_roller", "character_sheet", "spell_casting", 
                "skill_check", "rule_lookup"
            ],
            DnDRole.PLAYER_ROGUE: [
                "dice_roller", "character_sheet", "skill_check", 
                "inventory_management"
            ],
            DnDRole.PLAYER_CLERIC: [
                "dice_roller", "character_sheet", "spell_casting", 
                "skill_check"
            ]
        }
        
        # Role-specific rules and capabilities
        role_rules = {
            DnDRole.DUNGEON_MASTER: {
                "authority": "narrative_control",
                "responsibilities": ["world_consistency", "npc_portrayal", "rule_arbitration"],
                "creative_freedom": "high"
            },
            DnDRole.PLAYER_FIGHTER: {
                "authority": "character_actions",
                "specialties": ["combat", "physical_challenges", "leadership"],
                "creative_freedom": "medium"
            },
            DnDRole.PLAYER_WIZARD: {
                "authority": "character_actions",
                "specialties": ["magic", "knowledge", "problem_solving"],
                "creative_freedom": "medium"
            },
            DnDRole.PLAYER_ROGUE: {
                "authority": "character_actions", 
                "specialties": ["stealth", "investigation", "social_manipulation"],
                "creative_freedom": "medium"
            },
            DnDRole.PLAYER_CLERIC: {
                "authority": "character_actions",
                "specialties": ["healing", "divine_magic", "moral_guidance"],
                "creative_freedom": "medium"
            }
        }
        
        return AgentContext(
            world_state=self.world_state,
            available_tools=role_tools.get(agent_role, []),
            other_agents=self.get_other_agents(agent_role),
            scenario_rules=role_rules.get(agent_role, {}),
            time_constraints={"session_length": "4_hours", "turn_time": "flexible"}
        )
    
    def get_other_agents(self, current_role: DnDRole) -> List[str]:
        """Get list of other agents this role can interact with"""
        all_roles = [role.value for role in DnDRole if role != current_role]
        return all_roles
    
    def trigger_event(self, event_type: str, details: Dict[str, Any]):
        """Trigger D&D session events"""
        events = {
            "combat_encounter": self._handle_combat,
            "social_encounter": self._handle_social_interaction,
            "puzzle_challenge": self._handle_puzzle,
            "exploration": self._handle_exploration,
            "plot_twist": self._handle_plot_twist,
            "character_development": self._handle_character_moment,
            "rule_dispute": self._handle_rule_question
        }
        
        if event_type in events:
            result = events[event_type](details)
            self.world_state["world_events"].append({
                "type": event_type,
                "details": details,
                "result": result,
                "timestamp": "current_turn"
            })
            return result
    
    def _handle_combat(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle combat encounter"""
        self.world_state["current_challenge"] = "combat"
        return {
            "initiative_order": "roll_for_initiative",
            "enemy_stats": details.get("enemies", []),
            "battlefield": details.get("terrain", "open_ground"),
            "victory_conditions": "defeat_all_enemies"
        }
    
    def _handle_social_interaction(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle social encounter"""
        self.world_state["current_challenge"] = "social"
        npc_name = details.get("npc", "unknown")
        self.world_state["npc_reactions"][npc_name] = "neutral"
        return {
            "npc_personality": details.get("personality", "friendly"),
            "conversation_goals": details.get("goals", ["gather_information"]),
            "success_conditions": "persuasion_or_deception"
        }
    
    def _handle_puzzle(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle puzzle challenge"""
        self.world_state["current_challenge"] = "puzzle"
        return {
            "puzzle_type": details.get("type", "riddle"),
            "clues_available": details.get("clues", []),
            "time_pressure": details.get("timed", False),
            "success_reward": details.get("reward", "progress")
        }
    
    def _handle_exploration(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exploration phase"""
        location = details.get("location", "unknown")
        self.world_state["location"] = location
        return {
            "discoveries": details.get("discoveries", []),
            "hidden_elements": details.get("secrets", []),
            "environmental_challenges": details.get("hazards", [])
        }
    
    def _handle_plot_twist(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plot twist revelation"""
        twist = details.get("revelation", "unexpected_truth")
        self.world_state["story_progress"]["plot_twists"] = self.world_state["story_progress"].get("plot_twists", [])
        self.world_state["story_progress"]["plot_twists"].append(twist)
        return {
            "narrative_impact": "high",
            "player_reactions": "surprise",
            "story_implications": details.get("implications", [])
        }
    
    def _handle_character_moment(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle character development moment"""
        character = details.get("character", "unknown")
        return {
            "character_growth": details.get("growth_type", "backstory_reveal"),
            "emotional_impact": details.get("impact", "medium"),
            "party_bonding": "increased"
        }
    
    def _handle_rule_question(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rule interpretation question"""
        return {
            "rule_reference": details.get("rule", "unknown"),
            "dm_ruling": details.get("ruling", "dm_discretion"),
            "table_consensus": details.get("agreement", "accepted")
        }
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get metrics for D&D session analysis"""
        return {
            "player_engagement": {
                "participation_rate": "measure_speaking_time",
                "creative_contributions": "count_creative_solutions",
                "rule_interactions": "track_rule_usage",
                "character_roleplay": "assess_character_consistency"
            },
            "narrative_quality": {
                "story_coherence": "evaluate_plot_consistency",
                "world_building": "assess_setting_detail",
                "character_development": "track_growth_arcs",
                "pacing": "analyze_scene_transitions"
            },
            "game_mechanics": {
                "rule_adherence": "track_rule_following",
                "dice_roll_frequency": len(self.world_state.get("dice_rolls", [])),
                "combat_efficiency": "measure_combat_duration",
                "problem_solving": "analyze_solution_creativity"
            },
            "social_dynamics": {
                "party_cohesion": self.world_state["party_status"]["cohesion"],
                "leadership_patterns": "identify_decision_makers",
                "conflict_resolution": "track_disagreement_handling",
                "collaborative_storytelling": "measure_shared_narrative"
            },
            "dm_performance": {
                "adaptability": "measure_improvisation",
                "fairness": "assess_rule_consistency",
                "creativity": "evaluate_scenario_originality",
                "player_satisfaction": "gauge_enjoyment_levels"
            }
        }
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate summary of D&D session"""
        return {
            "session_highlights": [
                event for event in self.world_state["world_events"]
                if event["type"] in ["combat_encounter", "plot_twist", "character_development"]
            ],
            "story_progress": self.world_state["story_progress"],
            "character_moments": [
                event for event in self.world_state["world_events"]
                if event["type"] == "character_development"
            ],
            "memorable_quotes": "extract_from_dialogue",
            "next_session_hooks": "identify_unresolved_threads"
        }

# Example configuration for D&D session
DND_CONFIG = {
    "scenario_name": "the_lost_artifact",
    "campaign_setting": "forgotten_realms",
    "session_length": "4_hours",
    "party_size": 4,
    "experience_level": "intermediate",
    "focus_areas": [
        "creative_problem_solving",
        "character_roleplay",
        "collaborative_storytelling",
        "rule_interpretation"
    ],
    "analysis_metrics": [
        "player_engagement",
        "narrative_coherence", 
        "social_dynamics",
        "creative_solutions"
    ]
}