"""
Prison Simulation Scenario Template
Simulates life in a correctional facility with focus on:
- Social dynamics and power structures
- Rehabilitation programs effectiveness
- Behavioral adaptation patterns
- Intervention strategies
"""

from core.interfaces.agent_interface import AgentContext, AgentCapability
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import random

class PrisonRole(Enum):
    INMATE_LEADER = "inmate_leader"
    NEW_INMATE = "new_inmate"
    VETERAN_INMATE = "veteran_inmate"
    GUARD_CAPTAIN = "guard_captain"
    GUARD = "guard"
    COUNSELOR = "counselor"
    WARDEN = "warden"
    VISITOR = "visitor"

@dataclass
class PrisonPersona:
    """Prison-specific persona traits"""
    sentence_length: int  # months
    crime_type: str
    time_served: int  # months
    behavioral_record: List[str]
    rehabilitation_programs: List[str]
    social_connections: List[str]
    psychological_profile: Dict[str, Any]

class PrisonSimulation:
    """Prison environment simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_state = {
            "time_of_day": "morning",
            "current_activity": "breakfast",
            "security_level": "normal",
            "recent_incidents": [],
            "active_programs": [],
            "contraband_level": "low",
            "tension_level": 3  # 1-10 scale
        }
        
        self.locations = {
            "cells": {"capacity": 200, "current": 180},
            "cafeteria": {"capacity": 100, "current": 0},
            "yard": {"capacity": 150, "current": 0},
            "library": {"capacity": 30, "current": 0},
            "gym": {"capacity": 50, "current": 0},
            "counseling_rooms": {"capacity": 10, "current": 0},
            "workshop": {"capacity": 40, "current": 0}
        }
        
        self.available_tools = [
            "communication_system",
            "incident_reporting",
            "program_enrollment",
            "visitor_scheduling",
            "disciplinary_actions",
            "psychological_assessment",
            "contraband_search",
            "lockdown_protocol"
        ]
    
    def get_scenario_context(self, agent_role: PrisonRole) -> AgentContext:
        """Generate context specific to agent role"""
        
        # Role-specific tool access
        role_tools = {
            PrisonRole.INMATE_LEADER: ["communication_system", "program_enrollment"],
            PrisonRole.NEW_INMATE: ["communication_system", "program_enrollment", "visitor_scheduling"],
            PrisonRole.GUARD_CAPTAIN: ["incident_reporting", "disciplinary_actions", "lockdown_protocol", "contraband_search"],
            PrisonRole.GUARD: ["incident_reporting", "disciplinary_actions", "contraband_search"],
            PrisonRole.COUNSELOR: ["psychological_assessment", "program_enrollment", "visitor_scheduling"],
            PrisonRole.WARDEN: ["lockdown_protocol", "disciplinary_actions", "incident_reporting"]
        }
        
        # Role-specific scenario rules
        role_rules = {
            PrisonRole.INMATE_LEADER: {
                "movement_restrictions": ["yard", "cafeteria", "cells", "library"],
                "interaction_limits": "can_influence_other_inmates",
                "authority_level": "high_among_inmates"
            },
            PrisonRole.NEW_INMATE: {
                "movement_restrictions": ["cells", "cafeteria"],
                "interaction_limits": "limited_social_connections",
                "authority_level": "none"
            },
            PrisonRole.GUARD_CAPTAIN: {
                "movement_restrictions": "unrestricted",
                "interaction_limits": "professional_only",
                "authority_level": "high"
            }
        }
        
        return AgentContext(
            world_state=self.world_state,
            available_tools=role_tools.get(agent_role, []),
            other_agents=self.get_other_agents(agent_role),
            scenario_rules=role_rules.get(agent_role, {}),
            time_constraints={"daily_schedule": self.get_daily_schedule()}
        )
    
    def get_other_agents(self, current_role: PrisonRole) -> List[str]:
        """Get list of other agents this role can interact with"""
        all_roles = [role.value for role in PrisonRole if role != current_role]
        return all_roles
    
    def get_daily_schedule(self) -> Dict[str, str]:
        """Prison daily schedule"""
        return {
            "06:00": "wake_up",
            "06:30": "breakfast",
            "08:00": "work_assignments",
            "12:00": "lunch",
            "13:00": "yard_time",
            "15:00": "programs_counseling",
            "17:00": "dinner",
            "18:00": "recreation",
            "21:00": "lockdown"
        }
    
    def trigger_event(self, event_type: str, details: Dict[str, Any]):
        """Trigger scenario events"""
        events = {
            "contraband_discovery": self._handle_contraband_discovery,
            "fight_incident": self._handle_fight_incident,
            "lockdown_protocol": self._handle_lockdown,
            "new_inmate_arrival": self._handle_new_arrival,
            "program_completion": self._handle_program_completion,
            "visitor_day": self._handle_visitor_day
        }
        
        if event_type in events:
            events[event_type](details)
            self.world_state["recent_incidents"].append({
                "type": event_type,
                "details": details,
                "timestamp": "current_time"
            })
    
    def _handle_contraband_discovery(self, details: Dict[str, Any]):
        """Handle contraband discovery event"""
        self.world_state["security_level"] = "heightened"
        self.world_state["tension_level"] = min(10, self.world_state["tension_level"] + 2)
        
    def _handle_fight_incident(self, details: Dict[str, Any]):
        """Handle fight incident"""
        self.world_state["tension_level"] = min(10, self.world_state["tension_level"] + 3)
        if self.world_state["tension_level"] > 7:
            self.world_state["security_level"] = "lockdown"
    
    def _handle_lockdown(self, details: Dict[str, Any]):
        """Handle lockdown protocol"""
        self.world_state["security_level"] = "lockdown"
        self.world_state["current_activity"] = "confined_to_cells"
        
    def _handle_new_arrival(self, details: Dict[str, Any]):
        """Handle new inmate arrival"""
        self.world_state["tension_level"] = min(10, self.world_state["tension_level"] + 1)
        
    def _handle_program_completion(self, details: Dict[str, Any]):
        """Handle program completion"""
        self.world_state["tension_level"] = max(1, self.world_state["tension_level"] - 1)
        
    def _handle_visitor_day(self, details: Dict[str, Any]):
        """Handle visitor day"""
        self.world_state["current_activity"] = "visitation"
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get metrics for behavioral analysis"""
        return {
            "social_dynamics": {
                "power_structures": "analyze_leadership_patterns",
                "group_formations": "track_social_connections",
                "influence_networks": "map_influence_relationships"
            },
            "behavioral_patterns": {
                "adaptation_rate": "measure_new_inmate_integration",
                "recidivism_indicators": "identify_risk_factors",
                "program_effectiveness": "track_rehabilitation_success"
            },
            "intervention_effectiveness": {
                "counseling_impact": "measure_behavioral_changes",
                "program_participation": "track_engagement_levels",
                "disciplinary_actions": "analyze_deterrent_effects"
            },
            "environmental_factors": {
                "tension_levels": self.world_state["tension_level"],
                "incident_frequency": len(self.world_state["recent_incidents"]),
                "security_responses": "track_protocol_effectiveness"
            }
        }

# Example configuration for prison simulation
PRISON_CONFIG = {
    "scenario_name": "medium_security_prison",
    "duration_days": 30,
    "population": 180,
    "staff_count": 45,
    "programs_available": [
        "anger_management",
        "substance_abuse_counseling", 
        "vocational_training",
        "education_classes",
        "therapy_sessions"
    ],
    "analysis_focus": [
        "power_dynamics",
        "behavioral_adaptation", 
        "intervention_effectiveness",
        "recidivism_prediction"
    ]
}