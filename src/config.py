"""Configuration module for the scholarship agent."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

CONFIG_DIR = Path(__file__).parent.parent / "config"
RESOURCES_DIR = Path(__file__).parent.parent / "resources"

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9
    do_sample: bool = True
    device_map: str = "auto"
    load_in_4bit: bool = True

@dataclass
class HumanizationConfig:
    level: str = "high"  # low, medium, high
    avoid_ai_patterns: bool = True
    use_wikipedia_analysis: bool = True
    variation_strength: float = 0.7
    max_retries: int = 3

@dataclass
class DiscoveryConfig:
    manual_targets_file: str = "targets.json"
    scrape_enabled: bool = True
    max_results_per_search: int = 10
    delay_between_requests: float = 2.0

@dataclass
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    humanization: HumanizationConfig = field(default_factory=HumanizationConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    resources_dir: Path = RESOURCES_DIR
    output_dir: Path = RESOURCES_DIR / "saved_responses"
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AgentConfig":
        """Load configuration from JSON file."""
        if config_path is None:
            config_path = CONFIG_DIR / "settings.json"
        
        config = cls()
        
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)
            
            if "model" in data:
                for k, v in data["model"].items():
                    if hasattr(config.model, k):
                        setattr(config.model, k, v)
            
            if "humanization" in data:
                for k, v in data["humanization"].items():
                    if hasattr(config.humanization, k):
                        setattr(config.humanization, k, v)
            
            if "discovery" in data:
                for k, v in data["discovery"].items():
                    if hasattr(config.discovery, k):
                        setattr(config.discovery, k, v)
        
        return config
    
    def save(self, config_path: Optional[Path] = None):
        """Save configuration to JSON file."""
        if config_path is None:
            config_path = CONFIG_DIR / "settings.json"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model": {
                "model_name": self.model.model_name,
                "max_length": self.model.max_length,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "load_in_4bit": self.model.load_in_4bit,
            },
            "humanization": {
                "level": self.humanization.level,
                "avoid_ai_patterns": self.humanization.avoid_ai_patterns,
                "use_wikipedia_analysis": self.humanization.use_wikipedia_analysis,
                "variation_strength": self.humanization.variation_strength,
            },
            "discovery": {
                "scrape_enabled": self.discovery.scrape_enabled,
                "max_results_per_search": self.discovery.max_results_per_search,
                "delay_between_requests": self.discovery.delay_between_requests,
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
