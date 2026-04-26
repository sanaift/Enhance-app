"""
app_enhanced.py — Enhanced Agentic AI Cybersecurity Framework for Smart Grid Threat Modelling
Enhancements:
  1. Defined threat taxonomy with classification & prediction (ThreatClassifierAgent)
  2. Multi-domain datasets: network, physical sensors, SCADA, DER
  3. Model performance metrics: precision, recall, F1, confusion matrix, detection time
"""

from __future__ import annotations

import copy
import logging
import math
import random
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import streamlit as st
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_score, recall_score, f1_score, accuracy_score
    )
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    IsolationForest = None
    RandomForestClassifier = None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================================ #
# THREAT TAXONOMY
# ============================================================================ #

THREAT_TYPES = {
    "normal":              {"label": "Normal Operation",         "severity": 0, "color": "#44cc44"},
    "false_data_injection":{"label": "False Data Injection (FDI)","severity": 3, "color": "#ff4444"},
    "replay_attack":       {"label": "Replay Attack",            "severity": 2, "color": "#ff8800"},
    "dos_attack":          {"label": "Denial-of-Service (DoS)",  "severity": 3, "color": "#ff4444"},
    "man_in_the_middle":   {"label": "Man-in-the-Middle (MitM)", "severity": 3, "color": "#ff4444"},
    "scada_command_injection": {"label": "SCADA Command Injection","severity": 4,"color": "#cc0000"},
    "physical_fault":      {"label": "Physical Fault / Overload","severity": 2, "color": "#ff8800"},
    "der_manipulation":    {"label": "DER Setpoint Manipulation","severity": 3, "color": "#ff4444"},
    "scanning_probe":      {"label": "Network Scanning / Probe", "severity": 1, "color": "#ffcc00"},
    "cascade_fault":       {"label": "Cascading Fault",          "severity": 4, "color": "#cc0000"},
}

THREAT_LABELS = list(THREAT_TYPES.keys())

# ============================================================================ #
# runtime_config.py
# ============================================================================ #

_DEFAULTS: Dict[str, Any] = {
    "behavioral_sensitivity": 3.0,
    "anomaly_contamination": 0.05,
    "cascade_propagation_threshold": 0.5,
    "strategy_exploration_rate": 0.2,
}

@dataclass
class RuntimeConfig:
    behavioral_sensitivity: float = field(default=_DEFAULTS["behavioral_sensitivity"])
    anomaly_contamination: float = field(default=_DEFAULTS["anomaly_contamination"])
    cascade_propagation_threshold: float = field(default=_DEFAULTS["cascade_propagation_threshold"])
    strategy_exploration_rate: float = field(default=_DEFAULTS["strategy_exploration_rate"])
    _change_log: list = field(default_factory=list, init=False, repr=False)

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        applied: Dict[str, Any] = {}
        for key, value in config_dict.items():
            if hasattr(self, key) and not key.startswith("_"):
                old_value = getattr(self, key)
                setattr(self, key, value)
                applied[key] = {"old": old_value, "new": value}
        if applied:
            self._change_log.append({"cycle": len(self._change_log), "changes": applied})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def reset_defaults(self) -> None:
        for key, value in _DEFAULTS.items():
            setattr(self, key, value)

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get_change_log(self) -> list:
        return list(self._change_log)

    def change_log_length(self) -> int:
        return len(self._change_log)

# ============================================================================ #
# base_agent.py
# ============================================================================ #

class BaseAgent(ABC):
    def __init__(self, agent_id: str, runtime_config: RuntimeConfig) -> None:
        self.agent_id = agent_id
        self.config: RuntimeConfig = runtime_config
        self._history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, Any] = {
            "total_calls": 0,
            "total_time_ms": 0.0,
            "last_result": None,
        }

    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        pass

    def _record(self, result: Dict[str, Any], elapsed_ms: float) -> None:
        self.performance_stats["total_calls"] += 1
        self.performance_stats["total_time_ms"] += elapsed_ms
        self.performance_stats["last_result"] = result
        self._history.append({"result": result, "elapsed_ms": elapsed_ms})
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        return self._history[-last_n:]

# ============================================================================ #
# LAYER 0: Multi-Domain Data Fusion
# ============================================================================ #

class DataFusionAgent(BaseAgent):
    """
    Fuses sensor readings from four data domains:
      - Network traffic features
      - Physical sensor readings (voltage, current, frequency)
      - SCADA telemetry (command sequences, register values)
      - DER (Distributed Energy Resource) setpoints and output
    Returns a unified feature vector and domain-specific sub-vectors.
    """
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        fused = {
            "agent_id": self.agent_id,
            "network": data.get("network", {}),
            "physical": data.get("physical", {}),
            "scada": data.get("scada", {}),
            "der": data.get("der", {}),
            "fused_features": (
                data.get("network", {}).get("features", []) +
                data.get("physical", {}).get("features", []) +
                data.get("scada", {}).get("features", []) +
                data.get("der", {}).get("features", [])
            ),
            "true_threat": data.get("true_threat", "normal"),
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record(fused, elapsed_ms)
        return fused

# ============================================================================ #
# LAYER 1: Perceptual agents
# ============================================================================ #

class BehavioralEnvelopeAgent(BaseAgent):
    def __init__(self, agent_id: str, runtime_config: RuntimeConfig, window_size: int = 50) -> None:
        super().__init__(agent_id=agent_id, runtime_config=runtime_config)
        self._window: deque = deque(maxlen=window_size)
        self.window_size = window_size

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        values: List[float] = self._extract_values(data)
        sigma_threshold: float = self.config.behavioral_sensitivity
        results = []
        for idx, value in enumerate(values):
            self._window.append(value)
            if len(self._window) < 5:
                continue
            arr = np.array(self._window)
            mean, std = arr.mean(), arr.std()
            if std == 0:
                continue
            z_score = abs((value - mean) / std)
            if z_score > sigma_threshold:
                results.append({
                    "index": idx, "value": value,
                    "z_score": round(z_score, 4),
                    "mean": round(mean, 4), "std": round(std, 4),
                })
        result = {
            "agent_id": self.agent_id, "anomalies": results,
            "anomaly_count": len(results),
            "threshold_sigma": sigma_threshold,
            "values_processed": len(values),
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record(result, elapsed_ms)
        return result

    @staticmethod
    def _extract_values(data: Any) -> List[float]:
        if isinstance(data, dict):
            raw = data.get("fused_features", data.get("values", data.get("value", [])))
        else:
            raw = data
        if isinstance(raw, (int, float)):
            return [float(raw)]
        return [float(v) for v in raw]


class AnomalyDetectionAgent(BaseAgent):
    def __init__(self, agent_id: str, runtime_config: RuntimeConfig) -> None:
        super().__init__(agent_id=agent_id, runtime_config=runtime_config)
        self._model = None
        self._last_contamination: Optional[float] = None
        self._training_buffer: List[List[float]] = []
        self._min_training_samples = 20

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        contamination: float = self.config.anomaly_contamination
        features = data.get("fused_features", data.get("features", []))
        if features and isinstance(features[0], (int, float)):
            features = [[float(v)] for v in features]
        else:
            features = [[float(x) for x in vec] for vec in features]

        if not features:
            result = {"agent_id": self.agent_id, "anomaly_count": 0, "anomalies": [], "model_ready": False}
            self._record(result, (time.perf_counter() - t0) * 1000.0)
            return result

        self._training_buffer.extend(features)
        if len(self._training_buffer) > 500:
            self._training_buffer = self._training_buffer[-500:]

        anomaly_indices: List[int] = []
        model_ready = False

        if _SKLEARN_AVAILABLE and len(self._training_buffer) >= self._min_training_samples:
            if self._model is None or contamination != self._last_contamination:
                safe_cont = max(0.01, min(0.49, contamination))
                self._model = IsolationForest(contamination=safe_cont, random_state=42, n_estimators=50)
                self._model.fit(self._training_buffer)
                self._last_contamination = contamination

            X = np.array(features)
            preds = self._model.predict(X)
            anomaly_indices = [i for i, p in enumerate(preds) if p == -1]
            model_ready = True

        result = {
            "agent_id": self.agent_id,
            "anomaly_count": len(anomaly_indices),
            "anomalies": anomaly_indices,
            "model_ready": model_ready,
            "contamination_used": contamination,
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record(result, elapsed_ms)
        return result

# ============================================================================ #
# THREAT CLASSIFIER AGENT (New)
# ============================================================================ #

class ThreatClassifierAgent(BaseAgent):
    """
    Classifies detected anomalies into specific threat types from the taxonomy.
    Uses a Random Forest trained on labelled feature vectors (simulated ground-truth).
    Also tracks confusion matrix data for performance reporting.
    """
    def __init__(self, agent_id: str, runtime_config: RuntimeConfig) -> None:
        super().__init__(agent_id=agent_id, runtime_config=runtime_config)
        self._classifier: Optional[Any] = None
        self._label_encoder = LabelEncoder() if _SKLEARN_AVAILABLE else None
        self._fitted = False
        # Labelled training buffer: (features, label)
        self._train_X: List[List[float]] = []
        self._train_y: List[str] = []
        self._min_samples_per_class = 5
        # Tracking for performance metrics
        self.y_true: List[str] = []
        self.y_pred: List[str] = []
        self.detection_times_ms: List[float] = []

    def add_labelled_sample(self, features: List[float], label: str) -> None:
        """Add a ground-truth labelled sample for training."""
        self._train_X.append(features)
        self._train_y.append(label)
        if len(self._train_X) > 2000:
            self._train_X = self._train_X[-2000:]
            self._train_y = self._train_y[-2000:]

    def _try_fit(self) -> bool:
        if not _SKLEARN_AVAILABLE:
            return False
        label_counts = {l: self._train_y.count(l) for l in set(self._train_y)}
        if len(label_counts) < 2 or min(label_counts.values()) < self._min_samples_per_class:
            return False
        X = np.array(self._train_X)
        y = self._label_encoder.fit_transform(self._train_y)
        self._classifier = RandomForestClassifier(
            n_estimators=50, random_state=42, class_weight="balanced"
        )
        self._classifier.fit(X, y)
        self._fitted = True
        return True

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        features = data.get("fused_features", [])
        true_threat = data.get("true_threat", "normal")
        anomaly_count = data.get("anomaly_count", 0)

        # Add labelled sample to training buffer
        if features:
            flat = features if isinstance(features[0], (int, float)) else [v for vec in features for v in vec]
            self.add_labelled_sample(flat, true_threat)

        # Try to fit/update classifier
        if not self._fitted:
            self._try_fit()

        predicted_threat = "normal"
        confidence = 0.0
        threat_probabilities: Dict[str, float] = {}

        if self._fitted and features and _SKLEARN_AVAILABLE:
            flat = features if isinstance(features[0], (int, float)) else [v for vec in features for v in vec]
            X = np.array([flat])
            pred_idx = self._classifier.predict(X)[0]
            predicted_threat = self._label_encoder.inverse_transform([pred_idx])[0]
            proba = self._classifier.predict_proba(X)[0]
            classes = self._label_encoder.inverse_transform(self._classifier.classes_)
            threat_probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba)}
            confidence = float(max(proba))
        elif anomaly_count == 0:
            predicted_threat = "normal"
            confidence = 0.9
        else:
            # Rule-based fallback
            predicted_threat = _rule_based_classify(data)
            confidence = 0.6

        self.y_true.append(true_threat)
        self.y_pred.append(predicted_threat)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.detection_times_ms.append(elapsed_ms)

        result = {
            "agent_id": self.agent_id,
            "predicted_threat": predicted_threat,
            "true_threat": true_threat,
            "confidence": round(confidence, 4),
            "threat_info": THREAT_TYPES.get(predicted_threat, THREAT_TYPES["normal"]),
            "threat_probabilities": threat_probabilities,
            "classifier_ready": self._fitted,
        }
        self._record(result, elapsed_ms)
        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        if len(self.y_true) < 10:
            return {"error": "Not enough predictions yet (need ≥10)"}
        labels_present = sorted(set(self.y_true + self.y_pred))
        try:
            acc = accuracy_score(self.y_true, self.y_pred)
            prec = precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
            rec = recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
            f1 = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(self.y_true, self.y_pred, labels=labels_present)
            per_class = classification_report(
                self.y_true, self.y_pred, labels=labels_present,
                output_dict=True, zero_division=0
            )
            avg_det_time = statistics.mean(self.detection_times_ms) if self.detection_times_ms else 0.0
            return {
                "accuracy": round(acc, 4),
                "precision_weighted": round(prec, 4),
                "recall_weighted": round(rec, 4),
                "f1_weighted": round(f1, 4),
                "confusion_matrix": cm.tolist(),
                "confusion_labels": labels_present,
                "per_class_metrics": per_class,
                "avg_detection_time_ms": round(avg_det_time, 4),
                "total_predictions": len(self.y_true),
            }
        except Exception as e:
            return {"error": str(e)}


def _rule_based_classify(data: Dict[str, Any]) -> str:
    """Heuristic threat classifier for fallback when RF is not ready."""
    net = data.get("network", {})
    phys = data.get("physical", {})
    scada = data.get("scada", {})
    der = data.get("der", {})

    packet_rate = net.get("packet_rate", 0)
    voltage_dev = abs(phys.get("voltage_deviation", 0))
    cmd_anomaly = scada.get("command_anomaly", False)
    der_setpoint_jump = abs(der.get("setpoint_delta", 0))

    if packet_rate > 800:
        return "dos_attack"
    if cmd_anomaly and scada.get("unauthorized", False):
        return "scada_command_injection"
    if der_setpoint_jump > 0.4:
        return "der_manipulation"
    if voltage_dev > 0.25:
        return "false_data_injection"
    if net.get("replay_flag", False):
        return "replay_attack"
    if net.get("scanning", False):
        return "scanning_probe"
    if phys.get("overload", False):
        return "physical_fault"
    return "normal"


# ============================================================================ #
# LAYER 2: Cognitive agents
# ============================================================================ #

class CascadePredictorAgent(BaseAgent):
    _DEFAULT_TOPOLOGY: Dict[str, List[Tuple[str, float]]] = {
        "Substation_A": [("Substation_B", 0.8), ("DER_Bus_1", 0.4)],
        "Substation_B": [("Feeder_1", 0.7), ("Feeder_2", 0.3)],
        "DER_Bus_1":    [("Feeder_2", 0.6), ("Load_Center_1", 0.5)],
        "Feeder_1":     [("Load_Center_1", 0.9)],
        "Feeder_2":     [("Load_Center_2", 0.2)],
        "Load_Center_1":[],
        "Load_Center_2":[],
    }

    def __init__(self, agent_id: str, runtime_config: RuntimeConfig,
                 topology: Optional[Dict[str, List[Tuple[str, float]]]] = None,
                 max_depth: int = 10) -> None:
        super().__init__(agent_id=agent_id, runtime_config=runtime_config)
        self.topology = topology if topology is not None else self._DEFAULT_TOPOLOGY
        self.max_depth = max_depth

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        threshold: float = self.config.cascade_propagation_threshold

        source_nodes: List[str] = data.get("source_nodes") or []
        if not source_nodes:
            anomaly_count = data.get("anomaly_result", {}).get("anomaly_count", 0)
            envelope_count = data.get("envelope_result", {}).get("anomaly_count", 0)
            if anomaly_count > 0 or envelope_count > 0:
                all_nodes = list(self.topology.keys())
                source_nodes = all_nodes[:max(1, anomaly_count + envelope_count)]

        affected: Set[str] = set(source_nodes)
        paths: List[List[str]] = [[n] for n in source_nodes]
        depth = 0

        queue = list(source_nodes)
        visited = set(source_nodes)
        while queue and depth < self.max_depth:
            next_queue = []
            for node in queue:
                for neighbor, weight in self.topology.get(node, []):
                    if neighbor not in visited and weight >= threshold:
                        visited.add(neighbor)
                        affected.add(neighbor)
                        next_queue.append(neighbor)
                        for path in paths:
                            if path[-1] == node:
                                paths.append(path + [neighbor])
            queue = next_queue
            if queue:
                depth += 1

        result = {
            "agent_id": self.agent_id,
            "affected_nodes": list(affected),
            "affected_count": len(affected),
            "cascade_depth": depth,
            "propagation_paths": paths[:10],
            "threshold_used": threshold,
            "source_nodes": source_nodes,
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record(result, elapsed_ms)
        return result


# ============================================================================ #
# LAYER 3: Strategic agents
# ============================================================================ #

STRATEGIES: List[str] = [
    "load_shedding", "voltage_regulation", "reroute_power_flow",
    "isolate_fault_section", "demand_response", "generator_redispatch",
    "reactive_compensation", "emergency_restart",
]

# Threat-to-strategy mapping for smarter selection
THREAT_STRATEGY_AFFINITY: Dict[str, List[str]] = {
    "false_data_injection":   ["isolate_fault_section", "voltage_regulation"],
    "replay_attack":          ["isolate_fault_section", "reroute_power_flow"],
    "dos_attack":             ["isolate_fault_section", "reroute_power_flow"],
    "man_in_the_middle":      ["isolate_fault_section", "emergency_restart"],
    "scada_command_injection":["emergency_restart", "isolate_fault_section"],
    "physical_fault":         ["load_shedding", "reactive_compensation"],
    "der_manipulation":       ["generator_redispatch", "demand_response"],
    "scanning_probe":         ["reroute_power_flow", "demand_response"],
    "cascade_fault":          ["load_shedding", "emergency_restart"],
    "normal":                 ["voltage_regulation", "demand_response"],
}


class MitigationGeneratorAgent(BaseAgent):
    def __init__(self, agent_id: str, runtime_config: RuntimeConfig,
                 strategies: Optional[List[str]] = None, seed: Optional[int] = None) -> None:
        super().__init__(agent_id=agent_id, runtime_config=runtime_config)
        self.strategies: List[str] = strategies or list(STRATEGIES)
        self._rng = random.Random(seed)
        self._q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: {s: 0.0 for s in self.strategies})
        self._selection_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {s: 0 for s in self.strategies})

    def update_strategy_reward(self, threat_profile: str, strategy: str, reward: float) -> None:
        if strategy in self._q_table[threat_profile]:
            n = self._selection_counts[threat_profile][strategy] + 1
            old_q = self._q_table[threat_profile][strategy]
            self._q_table[threat_profile][strategy] = old_q + (reward - old_q) / n
            self._selection_counts[threat_profile][strategy] = n

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        epsilon: float = self.config.strategy_exploration_rate
        predicted_threat = data.get("threat_result", {}).get("predicted_threat", "normal")
        threat_profile: str = self._build_threat_profile(data, predicted_threat)
        selected, mode = self._select_strategy(threat_profile, epsilon, predicted_threat)
        self._selection_counts[threat_profile][selected] += 1
        actions = self._build_action_plan(selected, data)

        q_vals = self._q_table[threat_profile]
        sorted_strats = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)[:3]

        result = {
            "agent_id": self.agent_id,
            "selected_strategy": selected,
            "selection_mode": mode,
            "exploration_rate": epsilon,
            "threat_profile": threat_profile,
            "predicted_threat": predicted_threat,
            "top_strategies": sorted_strats,
            "actions": actions,
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record(result, elapsed_ms)
        return result

    def _build_threat_profile(self, data: Dict[str, Any], predicted_threat: str) -> str:
        cascade = data.get("cascade_result", {})
        anomaly = data.get("anomaly_result", {})
        depth = cascade.get("cascade_depth", 0)
        severity = THREAT_TYPES.get(predicted_threat, {}).get("severity", 0)
        if severity >= 4 or depth >= 3:
            return "critical"
        elif severity >= 2 or depth >= 1:
            return "moderate"
        return "normal"

    def _select_strategy(self, profile: str, epsilon: float, threat_type: str) -> Tuple[str, str]:
        affinity = THREAT_STRATEGY_AFFINITY.get(threat_type, [])
        if self._rng.random() < epsilon:
            return self._rng.choice(self.strategies), "exploration"
        # Prefer affinity-matched strategies with Q-table guidance
        q_vals = self._q_table[profile]
        if affinity:
            affinity_q = {s: q_vals.get(s, 0.0) for s in affinity}
            best = max(affinity_q, key=affinity_q.get)
        else:
            best = max(q_vals, key=q_vals.get)
        return best, "exploitation"

    def _build_action_plan(self, strategy: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        cascade = data.get("cascade_result", {})
        affected = cascade.get("affected_nodes", [])
        plans = {
            "load_shedding":         [{"action": "shed_load", "nodes": affected, "priority": "high"}],
            "voltage_regulation":    [{"action": "adjust_voltage", "target": "±5%", "nodes": affected}],
            "reroute_power_flow":    [{"action": "reroute", "from_nodes": affected, "to_nodes": ["backup"]}],
            "isolate_fault_section": [{"action": "isolate", "nodes": affected, "method": "breaker_trip"}],
            "demand_response":       [{"action": "demand_curtailment", "percentage": 20}],
            "generator_redispatch":  [{"action": "redispatch", "increase": ["G1"], "decrease": ["G2"]}],
            "reactive_compensation": [{"action": "inject_reactive_power", "nodes": affected}],
            "emergency_restart":     [{"action": "controlled_shutdown", "then": "restart", "nodes": affected}],
        }
        return plans.get(strategy, [{"action": strategy}])


# ============================================================================ #
# Performance Monitor
# ============================================================================ #

class AgentPerformanceMonitor:
    def __init__(self, framework: Any) -> None:
        self.framework = framework
        self._cycle_records: List[Dict[str, Any]] = []

    def record_cycle(self, cycle_output: Dict[str, Any]) -> None:
        record = {
            "cycle": cycle_output.get("cycle"),
            "anomaly_count": cycle_output.get("anomaly", {}).get("anomaly_count", 0),
            "cascade_depth": cycle_output.get("cascade", {}).get("cascade_depth", 0),
            "strategy": cycle_output.get("mitigation", {}).get("selected_strategy", ""),
            "predicted_threat": cycle_output.get("threat", {}).get("predicted_threat", "normal"),
            "true_threat": cycle_output.get("threat", {}).get("true_threat", "normal"),
        }
        self._cycle_records.append(record)

    def get_cycle_records(self) -> List[Dict[str, Any]]:
        return list(self._cycle_records)

    def generate_report(self) -> Dict[str, Any]:
        if not self._cycle_records:
            return {}
        anom = [r["anomaly_count"] for r in self._cycle_records]
        casc = [r["cascade_depth"] for r in self._cycle_records]
        strats = [r["strategy"] for r in self._cycle_records]
        return {
            "total_cycles": len(self._cycle_records),
            "total_anomalies": sum(anom),
            "avg_anomalies_per_cycle": round(statistics.mean(anom), 3),
            "max_cascade_depth": max(casc),
            "strategy_distribution": {s: strats.count(s) for s in set(strats)},
        }


# ============================================================================ #
# AgenticFramework
# ============================================================================ #

class AgenticFramework:
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None) -> None:
        self.runtime_config = RuntimeConfig()
        if config_overrides:
            self.runtime_config.update_from_dict(config_overrides)
        self.data_fusion_agent        = DataFusionAgent("data_fusion", self.runtime_config)
        self.behavioral_envelope_agent= BehavioralEnvelopeAgent("behavioral_envelope", self.runtime_config)
        self.anomaly_detection_agent  = AnomalyDetectionAgent("anomaly_detection", self.runtime_config)
        self.threat_classifier_agent  = ThreatClassifierAgent("threat_classifier", self.runtime_config)
        self.cascade_predictor_agent  = CascadePredictorAgent("cascade_predictor", self.runtime_config)
        self.mitigation_generator_agent = MitigationGeneratorAgent("mitigation_generator", self.runtime_config)
        self.performance_monitor = AgentPerformanceMonitor(framework=self)
        self._cycle_count: int = 0

    def update_runtime_config(self, config_dict: Dict[str, Any]) -> None:
        self.runtime_config.update_from_dict(config_dict)

    def run_cycle(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        self._cycle_count += 1

        fused_result   = self.data_fusion_agent.process(sensor_data)
        envelope_result= self.behavioral_envelope_agent.process(fused_result)
        anomaly_result = self.anomaly_detection_agent.process(fused_result)

        threat_input = {
            **fused_result,
            "anomaly_count": anomaly_result.get("anomaly_count", 0),
        }
        threat_result = self.threat_classifier_agent.process(threat_input)

        cascade_result = self.cascade_predictor_agent.process({
            "sensor_data": sensor_data,
            "envelope_result": envelope_result,
            "anomaly_result": anomaly_result,
        })
        mitigation_result = self.mitigation_generator_agent.process({
            "cascade_result": cascade_result,
            "anomaly_result": anomaly_result,
            "envelope_result": envelope_result,
            "threat_result": threat_result,
        })

        cycle_output = {
            "cycle": self._cycle_count,
            "fused": fused_result,
            "envelope": envelope_result,
            "anomaly": anomaly_result,
            "threat": threat_result,
            "cascade": cascade_result,
            "mitigation": mitigation_result,
            "config_snapshot": self.runtime_config.snapshot(),
        }
        self.performance_monitor.record_cycle(cycle_output)
        return cycle_output

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def get_agents(self) -> List[Any]:
        return [
            self.data_fusion_agent,
            self.behavioral_envelope_agent,
            self.anomaly_detection_agent,
            self.threat_classifier_agent,
            self.cascade_predictor_agent,
            self.mitigation_generator_agent,
        ]


# ============================================================================ #
# Adaptive Controller
# ============================================================================ #

class LearningController:
    _LR_SENSITIVITY   = 0.05
    _LR_CONTAMINATION = 0.002
    _LR_CASCADE       = 0.02
    _LR_EXPLORATION   = 0.01

    def __init__(self) -> None:
        self._sensitivity: float = 3.0
        self._contamination: float = 0.05
        self._cascade_threshold: float = 0.5
        self._exploration_rate: float = 0.2
        self._cycle_count: int = 0

    def update(self, metrics: Dict[str, Any]) -> None:
        self._cycle_count += 1
        fp_rate = metrics.get("false_positive_rate", 0.0)
        fn_rate = metrics.get("false_negative_rate", 0.0)
        cascade_over = metrics.get("cascade_overestimate_rate", 0.0)
        mit_success  = metrics.get("mitigation_success_rate", 0.5)
        self._sensitivity = max(1.0, min(6.0, self._sensitivity + self._LR_SENSITIVITY * (fp_rate - fn_rate)))
        self._contamination = max(0.01, min(0.3, self._contamination - self._LR_CONTAMINATION * fp_rate + self._LR_CONTAMINATION * fn_rate))
        self._cascade_threshold = max(0.1, min(0.95, self._cascade_threshold + self._LR_CASCADE * cascade_over - self._LR_CASCADE * fn_rate))
        self._exploration_rate = max(0.02, min(0.5, self._exploration_rate - self._LR_EXPLORATION * mit_success + self._LR_EXPLORATION * (1 - mit_success)))

    def get_current_thresholds(self) -> Dict[str, float]:
        return {
            "behavioral_sensitivity": round(self._sensitivity, 4),
            "anomaly_contamination": round(self._contamination, 6),
            "cascade_propagation_threshold": round(self._cascade_threshold, 4),
            "strategy_exploration_rate": round(self._exploration_rate, 4),
        }


class AdaptiveController:
    def __init__(self, framework: Any, learning_controller: Optional[LearningController] = None) -> None:
        self.framework = framework
        self.learning_controller = learning_controller or LearningController()
        self._adaptation_count: int = 0

    def adapt(self, cycle_output: Dict[str, Any]) -> Dict[str, float]:
        metrics = self._extract_metrics(cycle_output)
        self.learning_controller.update(metrics)
        new_thresholds = self.learning_controller.get_current_thresholds()
        self.framework.runtime_config.update_from_dict(new_thresholds)
        self._adaptation_count += 1
        return new_thresholds

    @staticmethod
    def _extract_metrics(cycle_output: Dict[str, Any]) -> Dict[str, float]:
        envelope = cycle_output.get("envelope", {})
        anomaly  = cycle_output.get("anomaly", {})
        cascade  = cycle_output.get("cascade", {})
        threat   = cycle_output.get("threat", {})
        n_envelope  = envelope.get("anomaly_count", 0)
        n_anomaly   = anomaly.get("anomaly_count", 0)
        cascade_depth = cascade.get("cascade_depth", 0)
        predicted   = threat.get("predicted_threat", "normal")
        true_threat = threat.get("true_threat", "normal")
        total_flags = max(n_envelope + n_anomaly, 1)
        fp_rate   = max(0.0, 1.0 - cascade_depth / total_flags) if total_flags else 0.0
        fn_rate   = 0.1 if cascade_depth > 0 and total_flags == 0 else 0.0
        cascade_over = 0.3 if cascade_depth == 0 and total_flags > 2 else 0.0
        mit_success  = 0.8 if predicted == true_threat else 0.3
        return {
            "false_positive_rate": round(fp_rate, 4),
            "false_negative_rate": round(fn_rate, 4),
            "cascade_overestimate_rate": round(cascade_over, 4),
            "mitigation_success_rate": mit_success,
        }

    @property
    def adaptation_count(self) -> int:
        return self._adaptation_count


# ============================================================================ #
# MULTI-DOMAIN DATA GENERATOR
# ============================================================================ #

def _multi_domain_data_generator(
    attack_probability: float = 0.20,
    seed: int = 42,
    t_offset: int = 0,
) -> Callable[[], Dict[str, Any]]:
    """
    Generates multi-domain smart grid sensor data with labelled threats.
    Domains: network traffic, physical sensors, SCADA, DER.
    seed=0 means fully random (uses system time), ensuring different results each run.
    t_offset shifts the phase of periodic signals so repeated runs look different.
    """
    actual_seed = seed if seed != 0 else int(time.time() * 1000) % 999999
    rng = random.Random(actual_seed)
    _t = [t_offset]

    def _generate() -> Dict[str, Any]:
        t = _t[0]
        _t[0] += 1

        # --- Decide true threat ---
        if t < 10 or rng.random() > attack_probability:
            true_threat = "normal"
        else:
            true_threat = rng.choice([
                "false_data_injection", "replay_attack", "dos_attack",
                "man_in_the_middle", "scada_command_injection",
                "physical_fault", "der_manipulation", "scanning_probe",
                "cascade_fault",
            ])

        severity = THREAT_TYPES[true_threat]["severity"]

        # ---- NETWORK domain ----
        base_packet_rate = 120 + 30 * math.sin(2 * math.pi * t / 40)
        packet_rate = base_packet_rate + rng.gauss(0, 5)
        latency_ms  = 8 + rng.gauss(0, 1)
        packet_loss = max(0, rng.gauss(0.002, 0.001))
        syn_ratio   = 0.05 + rng.gauss(0, 0.005)
        replay_flag = False
        scanning    = False

        if true_threat == "dos_attack":
            packet_rate += rng.uniform(700, 1200)
            syn_ratio   += rng.uniform(0.3, 0.6)
            latency_ms  += rng.uniform(50, 200)
        elif true_threat == "replay_attack":
            replay_flag  = True
            packet_rate += rng.uniform(20, 80)
        elif true_threat == "man_in_the_middle":
            latency_ms  += rng.uniform(30, 100)
            packet_loss += rng.uniform(0.01, 0.05)
        elif true_threat == "scanning_probe":
            scanning     = True
            packet_rate += rng.uniform(50, 150)
            syn_ratio   += rng.uniform(0.1, 0.3)

        network = {
            "packet_rate": round(packet_rate, 2),
            "latency_ms": round(latency_ms, 3),
            "packet_loss": round(packet_loss, 5),
            "syn_ratio": round(syn_ratio, 5),
            "replay_flag": replay_flag,
            "scanning": scanning,
            "features": [packet_rate / 1000, latency_ms / 200, packet_loss * 100, syn_ratio * 10],
        }

        # ---- PHYSICAL domain ----
        voltage    = 1.0 + 0.02 * math.sin(2 * math.pi * t / 20) + rng.gauss(0, 0.005)
        current    = 0.8 + 0.1 * math.sin(2 * math.pi * t / 15) + rng.gauss(0, 0.01)
        frequency  = 50.0 + rng.gauss(0, 0.02)
        temperature= 35   + 5 * math.sin(2 * math.pi * t / 100) + rng.gauss(0, 0.5)
        overload   = False
        voltage_deviation = 0.0

        if true_threat == "false_data_injection":
            voltage    += rng.choice([-1, 1]) * rng.uniform(0.25, 0.5)
            voltage_deviation = abs(voltage - 1.0)
        elif true_threat == "physical_fault":
            overload   = True
            current   += rng.uniform(0.5, 1.5)
            temperature+= rng.uniform(20, 50)
        elif true_threat == "cascade_fault":
            voltage   -= rng.uniform(0.3, 0.6)
            current   += rng.uniform(0.3, 0.8)
            frequency += rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)

        physical = {
            "voltage": round(voltage, 4),
            "current": round(current, 4),
            "frequency": round(frequency, 4),
            "temperature_c": round(temperature, 2),
            "voltage_deviation": round(voltage_deviation, 4),
            "overload": overload,
            "features": [voltage, current, frequency / 50, temperature / 100],
        }

        # ---- SCADA domain ----
        normal_cmd_id = rng.choice([1, 2, 3, 4, 5])
        register_val  = 100 + 10 * math.sin(2 * math.pi * t / 30) + rng.gauss(0, 1)
        command_anomaly= False
        unauthorized   = False

        if true_threat == "scada_command_injection":
            normal_cmd_id  = rng.randint(90, 255)   # unusual command code
            command_anomaly= True
            unauthorized   = True
            register_val  += rng.choice([-1, 1]) * rng.uniform(40, 100)
        elif true_threat == "replay_attack":
            command_anomaly= True

        scada = {
            "command_id": normal_cmd_id,
            "register_value": round(register_val, 3),
            "command_anomaly": command_anomaly,
            "unauthorized": unauthorized,
            "features": [normal_cmd_id / 255, register_val / 200, int(command_anomaly), int(unauthorized)],
        }

        # ---- DER domain ----
        pv_output   = max(0, 0.6 + 0.3 * math.sin(2 * math.pi * t / 50) + rng.gauss(0, 0.02))
        wind_output = max(0, 0.4 + 0.2 * math.sin(2 * math.pi * t / 35 + 1) + rng.gauss(0, 0.03))
        setpoint    = 0.5 + rng.gauss(0, 0.01)
        setpoint_delta = 0.0
        battery_soc = 0.7 + 0.1 * math.sin(2 * math.pi * t / 200) + rng.gauss(0, 0.005)

        if true_threat == "der_manipulation":
            setpoint_delta = rng.choice([-1, 1]) * rng.uniform(0.4, 0.8)
            setpoint  += setpoint_delta
            pv_output += rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)

        der = {
            "pv_output_pu": round(max(0, pv_output), 4),
            "wind_output_pu": round(max(0, wind_output), 4),
            "setpoint": round(setpoint, 4),
            "setpoint_delta": round(setpoint_delta, 4),
            "battery_soc": round(min(1.0, max(0.0, battery_soc)), 4),
            "features": [pv_output, wind_output, setpoint, battery_soc],
        }

        # Legacy fields for backward compat
        all_values = network["features"] + physical["features"] + scada["features"] + der["features"]
        return {
            "timestamp": t,
            "true_threat": true_threat,
            "network":  network,
            "physical": physical,
            "scada":    scada,
            "der":      der,
            "values":   all_values,
            "features": [[v] for v in all_values],
            "source_nodes": [],
        }
    return _generate


# ============================================================================ #
# Streamlit UI
# ============================================================================ #

st.set_page_config(
    page_title="Smart Grid Cyber Threat Framework",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .stMetric { background: #0f1117; border-radius: 8px; padding: 8px; }
    .threat-high { color: #ff4444; font-weight: bold; }
    .threat-med  { color: #ffa500; font-weight: bold; }
    .threat-low  { color: #44cc44; font-weight: bold; }
    .agent-card {
        background: #FFFFFF;
        border: 1px solid #D0D8E8;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #2E75B6;
        color: #1A1A1A;
        font-family: Arial, sans-serif;
    }
    .agent-card .agent-name { font-size: 15px; font-weight: bold; color: #1F4E79; }
    .agent-card .agent-layer { font-size: 12px; color: #2E75B6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .agent-card .agent-role { font-size: 12px; color: #444; margin-bottom: 6px; font-style: italic; }
    .agent-card .agent-stats { font-size: 13px; color: #222; }
    .agent-card .stat-badge { display: inline-block; background: #EBF3FB; color: #1F4E79; border-radius: 4px; padding: 2px 8px; margin-right: 6px; font-weight: 600; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Agentic AI Cybersecurity Framework — Smart Grid Threat Modelling")
st.caption("4-layer hierarchical multi-agent system: Data Fusion → Perceptual → Cognitive → Strategic | Multi-domain datasets | Threat classification & performance metrics")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Framework Configuration")
    sens = st.slider("Behavioral Sensitivity (σ)", 1.0, 6.0, 3.0, 0.1)
    cont = st.slider("Anomaly Contamination", 0.01, 0.30, 0.05, 0.01)
    casc = st.slider("Cascade Propagation Threshold", 0.1, 0.95, 0.5, 0.05)
    expl = st.slider("Strategy Exploration Rate (ε)", 0.02, 0.5, 0.2, 0.01)
    st.divider()
    attack_prob = st.slider("Attack Probability", 0.05, 0.60, 0.25, 0.05,
                            help="Fraction of cycles with a simulated attack")
    n_cycles = st.number_input("Simulation Cycles", 50, 500, 150, step=50)
    use_adaptive = st.checkbox("Enable Adaptive Controller", value=True)
    seed = st.number_input("Random Seed", 0, 9999, 42)
    st.divider()
    st.subheader("📂 Dataset Source")
    dataset_mode = st.radio(
        "Choose data source:",
        ["🔧 Synthetic (built-in generator)",
         "📁 Dataset 1 — Smart Grid Synthetic (Custom CSV)",
         "📡 Dataset 2 — TON_IoT Style (UNSW Sydney)"],
        index=0,
        help="Dataset 2 uses TON_IoT schema (Moustafa 2021) with different "
             "signal ranges, 60Hz grid, wind-dominant DER, and gas-pipeline SCADA."
    )
    uploaded_file = None
    if "Dataset 1" in dataset_mode or "Dataset 2" in dataset_mode:
        uploaded_file = st.file_uploader(
            "Upload CSV (columns: packet_rate, latency_ms, voltage, current, "
            "frequency, register_value, pv_output_pu, setpoint — optionally: true_threat)",
            type=["csv"],
            help="For Dataset 2, use the TON_IoT-style sample CSV provided."
        )
        if uploaded_file is not None:
            st.success(f"✅ Loaded: {uploaded_file.name}")
        else:
            if "Dataset 2" in dataset_mode:
                st.info("Upload the **ton_iot_smartgrid_dataset.csv** file above.")
            else:
                st.info("Upload the **sample_smartgrid_dataset.csv** file above.")
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ---- State ----
for k in ["results", "fw", "ctrl", "threshold_history"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ────────────────────────────────────────────────────────────────────────────
# CSV Dataset Parser
# ────────────────────────────────────────────────────────────────────────────
EXPECTED_COLS = {
    "network": ["packet_rate", "latency_ms", "packet_loss", "syn_ratio"],
    "physical": ["voltage", "current", "frequency", "temperature_c"],
    "scada":   ["register_value", "command_id"],
    "der":     ["pv_output_pu", "wind_output_pu", "setpoint", "battery_soc"],
}

def _parse_uploaded_csv(df: pd.DataFrame) -> list:
    """
    Convert an uploaded DataFrame into the dict format expected by AgenticFramework.run_cycle().
    Missing columns are filled with domain-appropriate defaults.
    Supports flexible column names (case-insensitive, underscore/space tolerant).
    """
    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    records = []
    for i, row in df.iterrows():
        def g(col, default=0.0):
            """Get column value with fallback."""
            return float(row[col]) if col in row.index else default

        # Network
        packet_rate = g("packet_rate", 120.0)
        latency_ms  = g("latency_ms",  8.0)
        packet_loss = g("packet_loss", 0.002)
        syn_ratio   = g("syn_ratio",   0.05)
        net_feats   = [packet_rate / 1000, latency_ms / 200, packet_loss * 100, syn_ratio * 10]

        # Physical
        voltage     = g("voltage",     1.0)
        current     = g("current",     0.8)
        frequency   = g("frequency",   50.0)
        temperature = g("temperature_c", 35.0)
        phys_feats  = [voltage, current, frequency / 50, temperature / 100]

        # SCADA
        reg_val     = g("register_value", 100.0)
        cmd_id      = g("command_id",     2.0)
        scada_feats = [cmd_id / 255, reg_val / 200, 0.0, 0.0]

        # DER
        pv_out      = g("pv_output_pu",  0.5)
        wind_out    = g("wind_output_pu",0.4)
        setpt       = g("setpoint",      0.5)
        batt_soc    = g("battery_soc",   0.7)
        der_feats   = [pv_out, wind_out, setpt, batt_soc]

        true_threat = str(row["true_threat"]).strip() if "true_threat" in row.index else "normal"
        if true_threat not in THREAT_LABELS:
            true_threat = "normal"

        all_values = net_feats + phys_feats + scada_feats + der_feats
        records.append({
            "timestamp":   i,
            "true_threat": true_threat,
            "network":  {"packet_rate": packet_rate, "latency_ms": latency_ms,
                         "packet_loss": packet_loss, "syn_ratio": syn_ratio,
                         "replay_flag": False, "scanning": False, "features": net_feats},
            "physical": {"voltage": voltage, "current": current, "frequency": frequency,
                         "temperature_c": temperature, "voltage_deviation": abs(voltage - 1.0),
                         "overload": current > 1.5, "features": phys_feats},
            "scada":    {"command_id": int(cmd_id), "register_value": reg_val,
                         "command_anomaly": False, "unauthorized": False, "features": scada_feats},
            "der":      {"pv_output_pu": pv_out, "wind_output_pu": wind_out,
                         "setpoint": setpt, "setpoint_delta": 0.0,
                         "battery_soc": batt_soc, "features": der_feats},
            "values":   all_values,
            "features": [[v] for v in all_values],
            "source_nodes": [],
        })
    return records

# ---- Run Simulation ----
if run_btn:
    # Seed numpy for IsolationForest reproducibility.
    # The data generator uses its own Random(seed) instance + t_offset
    # so results vary each run while remaining reproducible when seed != 0.
    np.random.seed(seed if seed != 0 else None)

    fw = AgenticFramework(config_overrides={
        "behavioral_sensitivity": sens,
        "anomaly_contamination": cont,
        "cascade_propagation_threshold": casc,
        "strategy_exploration_rate": expl,
    })
    ctrl = AdaptiveController(framework=fw) if use_adaptive else None
    # ── DATA SOURCE: uploaded CSV or synthetic generator ────────────────────
    use_uploaded = uploaded_file is not None
    if use_uploaded:
        try:
            csv_df = pd.read_csv(uploaded_file)
            dataset_records = _parse_uploaded_csv(csv_df)
            total_cycles = len(dataset_records)
            st.sidebar.success(f"Using uploaded dataset: {total_cycles} rows")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            st.stop()
    else:
        # Synthetic generator: t_offset makes every run produce different signals
        # even when the same seed is chosen, because the sine wave phase shifts.
        t_offset = int(time.time()) % 10000
        data_gen = _multi_domain_data_generator(
            attack_probability=attack_prob,
            seed=seed,
            t_offset=t_offset,
        )
        total_cycles = int(n_cycles)

    cycle_outputs = []
    threshold_history = []

    progress = st.progress(0, text="Running simulation...")
    for i in range(total_cycles):
        sensor_data = dataset_records[i] if use_uploaded else data_gen()
        out = fw.run_cycle(sensor_data)
        cycle_outputs.append(out)
        threshold_history.append(fw.runtime_config.snapshot())
        if ctrl:
            ctrl.adapt(out)
        progress.progress((i + 1) / total_cycles, text=f"Cycle {i+1}/{total_cycles}")

    progress.empty()
    st.session_state.results = cycle_outputs
    st.session_state.threshold_history = threshold_history
    st.session_state.fw = fw
    st.session_state.ctrl = ctrl

# ---- Display Results ----
if st.session_state.results:
    results = st.session_state.results
    fw      = st.session_state.fw
    th_hist = st.session_state.threshold_history

    # KPIs
    st.subheader("📊 Simulation Summary")
    if uploaded_file is not None:
        ds_name = uploaded_file.name
        if "ton_iot" in ds_name.lower() or "Dataset 2" in dataset_mode:
            data_source_label = f"📡 Dataset 2 — TON_IoT Style (UNSW Sydney 2021): {ds_name}"
            dataset_id = "DS2"
        else:
            data_source_label = f"📁 Dataset 1 — Smart Grid Custom CSV: {ds_name}"
            dataset_id = "DS1"
    else:
        data_source_label = "🔧 Synthetic Multi-Domain Generator (built-in)"
        dataset_id = "SYN"
    st.info(f"**Data source:** {data_source_label}  |  **Domains:** Network · Physical · SCADA · DER  |  **Features:** 16 total (4 per domain)")
    st.session_state["dataset_id"] = dataset_id
    anomaly_counts  = [r["anomaly"]["anomaly_count"] for r in results]
    cascade_depths  = [r["cascade"]["cascade_depth"] for r in results]
    affected_counts = [r["cascade"]["affected_count"] for r in results]
    strategies      = [r["mitigation"]["selected_strategy"] for r in results]
    threats_pred    = [r["threat"]["predicted_threat"] for r in results]
    threats_true    = [r["threat"]["true_threat"] for r in results]

    correct = sum(p == t for p, t in zip(threats_pred, threats_true))
    threat_acc = correct / len(threats_pred) if threats_pred else 0

    col1,col2,col3,col4,col5,col6 = st.columns(6)
    col1.metric("Total Cycles",           len(results))
    col2.metric("Total Anomalies",        sum(anomaly_counts))
    col3.metric("Max Cascade Depth",      max(cascade_depths))
    col4.metric("Max Affected Nodes",     max(affected_counts))
    col5.metric("Attack Cycles",          sum(1 for t in threats_true if t != "normal"))
    col6.metric("Threat Class. Accuracy", f"{threat_acc:.1%}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Detection",
        "🔍 Threat Classification",
        "📊 Model Performance",
        "🌐 Cascade Analysis",
        "🛡️ Mitigation",
        "🔄 Adaptive Learning",
        "🤖 Agent Status",
    ])

    # ---- TAB 1: Detection ----
    with tab1:
        st.markdown("""
<div style="background:#EBF3FB;border-left:5px solid #2E75B6;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#2E75B6;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Layer 1 — Perceptual</span><br>
<span style="font-size:15px;font-weight:bold;color:#1F4E79;">BehavioralEnvelopeAgent</span>
<span style="color:#555;font-size:13px;"> — Rolling σ-based deviation detector (window=50)</span>&nbsp;&nbsp;
<span style="font-size:15px;font-weight:bold;color:#1F4E79;">AnomalyDetectionAgent</span>
<span style="color:#555;font-size:13px;"> — IsolationForest unsupervised ML outlier detection</span>
</div>""", unsafe_allow_html=True)

        st.subheader("Anomaly Detection Over Time")
        cycles_ax = list(range(1, len(results) + 1))
        env_vals  = [r["envelope"]["anomaly_count"] for r in results]
        ml_vals   = [r["anomaly"]["anomaly_count"]  for r in results]
        if _PLOTLY:
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(x=cycles_ax, y=env_vals, mode="lines",
                name="Envelope Anomalies (BehavioralEnvelopeAgent)",
                line=dict(color="#E63946", width=2)))
            fig_anom.add_trace(go.Scatter(x=cycles_ax, y=ml_vals, mode="lines",
                name="ML Anomalies (AnomalyDetectionAgent)",
                line=dict(color="#2A9D8F", width=2, dash="dot")))
            fig_anom.update_layout(xaxis_title="Cycle", yaxis_title="Anomaly Count",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=40,r=20,t=30,b=60), height=320,
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            df_anom = pd.DataFrame({"Cycle": cycles_ax,
                "Envelope Anomalies": env_vals, "ML Anomalies": ml_vals})
            st.line_chart(df_anom.set_index("Cycle"))

        df_sensors = pd.DataFrame([{
            "Cycle":         r["cycle"],
            "Packet Rate":   r["fused"]["network"].get("packet_rate", 0),
            "Latency (ms)":  r["fused"]["network"].get("latency_ms", 0),
            "Voltage (pu)":  r["fused"]["physical"].get("voltage", 0),
            "Current (pu)":  r["fused"]["physical"].get("current", 0),
            "Frequency (Hz)":r["fused"]["physical"].get("frequency", 0),
            "PV Output":     r["fused"]["der"].get("pv_output_pu", 0),
            "SCADA Reg Val": r["fused"]["scada"].get("register_value", 0),
        } for r in results])

        if _PLOTLY:
            st.subheader("Physical Domain — Voltage, Current & Frequency")
            fig_phys = go.Figure()
            fig_phys.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["Voltage (pu)"],
                name="Voltage (pu)", line=dict(color="#E63946", width=2)))
            fig_phys.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["Current (pu)"],
                name="Current (pu)", line=dict(color="#F4A261", width=2)))
            fig_phys.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["Frequency (Hz)"] / 50,
                name="Frequency /50 (normalised)", line=dict(color="#2A9D8F", width=2, dash="dash")))
            fig_phys.update_layout(xaxis_title="Cycle", yaxis_title="Value",
                legend=dict(orientation="h",y=-0.3),
                margin=dict(l=40,r=20,t=30,b=70), height=300,
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
            st.plotly_chart(fig_phys, use_container_width=True)

            st.subheader("Network Domain — Traffic Features")
            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["Packet Rate"],
                name="Packet Rate (pkts/s)", line=dict(color="#264653", width=2)))
            fig_net.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["Latency (ms)"],
                name="Latency (ms)", line=dict(color="#E9C46A", width=2, dash="dot"),
                yaxis="y2"))
            fig_net.update_layout(
                xaxis_title="Cycle",
                yaxis=dict(title="Packet Rate", color="#264653"),
                yaxis2=dict(title="Latency (ms)", overlaying="y", side="right", color="#E9C46A"),
                legend=dict(orientation="h",y=-0.3),
                margin=dict(l=40,r=60,t=30,b=70), height=300,
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
            st.plotly_chart(fig_net, use_container_width=True)

            st.subheader("DER & SCADA Domains")
            fig_der = go.Figure()
            fig_der.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["PV Output"],
                name="PV Output (pu)", line=dict(color="#E76F51", width=2)))
            fig_der.add_trace(go.Scatter(x=df_sensors["Cycle"], y=df_sensors["SCADA Reg Val"],
                name="SCADA Register Value", line=dict(color="#6A0572", width=2, dash="dash"),
                yaxis="y2"))
            fig_der.update_layout(
                xaxis_title="Cycle",
                yaxis=dict(title="PV Output (pu)", color="#E76F51"),
                yaxis2=dict(title="SCADA Register", overlaying="y", side="right", color="#6A0572"),
                legend=dict(orientation="h",y=-0.3),
                margin=dict(l=40,r=60,t=30,b=70), height=300,
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
            st.plotly_chart(fig_der, use_container_width=True)
        else:
            st.line_chart(df_sensors.set_index("Cycle")[["Voltage (pu)", "Current (pu)", "Frequency (Hz)"]])
            st.line_chart(df_sensors.set_index("Cycle")[["Packet Rate", "Latency (ms)"]])
            st.line_chart(df_sensors.set_index("Cycle")[["PV Output", "SCADA Reg Val"]])

    # ---- TAB 2: Threat Classification ----
    with tab2:
        st.markdown("""
<div style="background:#EDF7ED;border-left:5px solid #375623;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#375623;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Layer 2 — Cognitive</span><br>
<span style="font-size:15px;font-weight:bold;color:#1F3319;">ThreatClassifierAgent</span>
<span style="color:#555;font-size:13px;"> — Random Forest supervised classifier (10 threat classes, balanced weights, online learning)</span>
</div>""", unsafe_allow_html=True)
        st.subheader("Threat Type Distribution")
        from collections import Counter
        true_dist = Counter(threats_true)
        pred_dist = Counter(threats_pred)
        all_threat_keys = sorted(set(list(true_dist.keys()) + list(pred_dist.keys())))
        df_dist = pd.DataFrame({
            "Threat Type": [THREAT_TYPES[k]["label"] for k in all_threat_keys],
            "True Count":  [true_dist.get(k, 0) for k in all_threat_keys],
            "Predicted Count": [pred_dist.get(k, 0) for k in all_threat_keys],
        })
        st.bar_chart(df_dist.set_index("Threat Type"))

        st.subheader("Threat Classification Timeline")
        df_timeline = pd.DataFrame({
            "Cycle":      range(1, len(results) + 1),
            "True Threat Index":  [THREAT_LABELS.index(t) if t in THREAT_LABELS else 0 for t in threats_true],
            "Pred Threat Index":  [THREAT_LABELS.index(t) if t in THREAT_LABELS else 0 for t in threats_pred],
        })
        st.line_chart(df_timeline.set_index("Cycle"))
        st.caption("Threat index mapping: " + " | ".join(f"{i}={k}" for i,k in enumerate(THREAT_LABELS)))

        st.subheader("Last Cycle Threat Details")
        last_threat = results[-1]["threat"]
        c1, c2, c3 = st.columns(3)
        info = last_threat.get("threat_info", {})
        c1.metric("Predicted Threat", info.get("label", last_threat["predicted_threat"]))
        c2.metric("True Threat", THREAT_TYPES.get(last_threat["true_threat"], {}).get("label", last_threat["true_threat"]))
        c3.metric("Confidence", f"{last_threat['confidence']:.1%}")

        probs = last_threat.get("threat_probabilities", {})
        if probs:
            df_probs = pd.DataFrame({"Threat": [THREAT_TYPES.get(k, {}).get("label", k) for k in probs], "Probability": list(probs.values())})
            df_probs = df_probs.sort_values("Probability", ascending=False)
            st.bar_chart(df_probs.set_index("Threat"))

        st.subheader("Threat Taxonomy")
        df_taxonomy = pd.DataFrame([
            {"ID": k, "Threat Name": v["label"], "Severity (0-4)": v["severity"]}
            for k, v in THREAT_TYPES.items()
        ])
        st.dataframe(df_taxonomy, use_container_width=True, hide_index=True)

    # ---- TAB 3: Model Performance ----
    with tab3:
        st.markdown("""
<div style="background:#EDF7ED;border-left:5px solid #375623;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#375623;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Layer 2 — Cognitive</span><br>
<span style="font-size:15px;font-weight:bold;color:#1F3319;">ThreatClassifierAgent</span>
<span style="color:#555;font-size:13px;"> — Real-time performance evaluation: Accuracy · Precision · Recall · F1 · Confusion Matrix · Detection Latency</span>
</div>""", unsafe_allow_html=True)
        st.subheader("🎯 Classifier Performance Metrics")
        metrics = fw.threat_classifier_agent.get_performance_metrics()

        if "error" in metrics:
            st.warning(metrics["error"])
        else:
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Accuracy",           f"{metrics['accuracy']:.3f}")
            c2.metric("Precision (W.Avg)", f"{metrics['precision_weighted']:.3f}")
            c3.metric("Recall (W.Avg)",    f"{metrics['recall_weighted']:.3f}")
            c4.metric("F1 Score (W.Avg)",  f"{metrics['f1_weighted']:.3f}")
            c5.metric("Avg Detection Time",f"{metrics['avg_detection_time_ms']:.2f} ms")

            st.subheader("Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            labels_short = [k[:10] for k in metrics["confusion_labels"]]
            df_cm = pd.DataFrame(cm, index=labels_short, columns=labels_short)
            st.dataframe(df_cm.style.background_gradient(cmap="YlOrRd"), use_container_width=True)
            st.caption("Rows = True label, Columns = Predicted label")

            st.subheader("Per-Class Metrics")
            per_class = metrics["per_class_metrics"]
            rows = []
            for key, vals in per_class.items():
                if key in ("accuracy", "macro avg", "weighted avg"):
                    continue
                if isinstance(vals, dict):
                    rows.append({
                        "Threat": THREAT_TYPES.get(key, {}).get("label", key),
                        "Precision": round(vals.get("precision", 0), 3),
                        "Recall":    round(vals.get("recall", 0), 3),
                        "F1-Score":  round(vals.get("f1-score", 0), 3),
                        "Support":   int(vals.get("support", 0)),
                    })
            if rows:
                df_per_class = pd.DataFrame(rows)
                st.dataframe(df_per_class, use_container_width=True, hide_index=True)

            st.subheader("Detection Time Distribution")
            det_times = fw.threat_classifier_agent.detection_times_ms
            if det_times:
                df_dt = pd.DataFrame({"Detection Time (ms)": det_times})
                st.line_chart(df_dt)
                col1, col2, col3 = st.columns(3)
                col1.metric("Min (ms)",  f"{min(det_times):.3f}")
                col2.metric("Mean (ms)", f"{statistics.mean(det_times):.3f}")
                col3.metric("Max (ms)",  f"{max(det_times):.3f}")

            st.subheader("Classifier Summary")
            st.info(f"""
**Total predictions:** {metrics['total_predictions']}  
**Classifier type:** Random Forest (50 trees, balanced class weights)  
**Feature domains:** Network (4 features) + Physical (4) + SCADA (4) + DER (4) = 16 features  
**Threat classes:** {len(THREAT_LABELS)} types  
**Note:** Classifier accuracy improves with more cycles as training data accumulates.
            """)

    # ---- TAB 4: Cascade Analysis ----
    with tab4:
        st.markdown("""
<div style="background:#EDF7ED;border-left:5px solid #375623;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#375623;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Layer 2 — Cognitive</span><br>
<span style="font-size:15px;font-weight:bold;color:#1F3319;">CascadePredictorAgent</span>
<span style="color:#555;font-size:13px;"> — BFS graph traversal over grid topology — predicts fault spread across substations, feeders and load centers</span>
</div>""", unsafe_allow_html=True)
        st.subheader("Cascade Propagation Over Time")
        df_casc = pd.DataFrame({
            "Cycle": range(1, len(results) + 1),
            "Cascade Depth":  cascade_depths,
            "Affected Nodes": affected_counts,
        })
        if _PLOTLY:
            fig_casc = go.Figure()
            fig_casc.add_trace(go.Scatter(x=df_casc["Cycle"], y=df_casc["Affected Nodes"],
                fill="tozeroy", name="Affected Nodes", line=dict(color="#E63946"),
                fillcolor="rgba(230,57,70,0.15)"))
            fig_casc.add_trace(go.Scatter(x=df_casc["Cycle"], y=df_casc["Cascade Depth"],
                fill="tozeroy", name="Cascade Depth", line=dict(color="#264653"),
                fillcolor="rgba(38,70,83,0.20)"))
            fig_casc.update_layout(xaxis_title="Cycle", yaxis_title="Count",
                legend=dict(orientation="h",y=-0.3),
                margin=dict(l=40,r=20,t=20,b=60), height=300,
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
            st.plotly_chart(fig_casc, use_container_width=True)
        else:
            st.area_chart(df_casc.set_index("Cycle"))

        st.subheader("Last Cycle Cascade Details")
        last_cascade = results[-1]["cascade"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cascade Depth",  last_cascade["cascade_depth"])
        c2.metric("Affected Nodes", last_cascade["affected_count"])
        c3.metric("Threshold Used", f"{last_cascade['threshold_used']:.3f}")

        if last_cascade["affected_nodes"]:
            st.write("**Affected Nodes:**", ", ".join(last_cascade["affected_nodes"]))
        if last_cascade["propagation_paths"]:
            st.write("**Propagation Paths (sample):**")
            for path in last_cascade["propagation_paths"][:3]:
                st.code(" → ".join(path))

    # ---- TAB 5: Mitigation ----
    with tab5:
        st.markdown("""
<div style="background:#FEF3E2;border-left:5px solid #843C0C;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#843C0C;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Layer 3 — Strategic</span><br>
<span style="font-size:15px;font-weight:bold;color:#5C2900;">MitigationGeneratorAgent</span>
<span style="color:#555;font-size:13px;"> — ε-greedy Q-Learning policy with threat-affinity mapping over 8 response strategies</span>
</div>""", unsafe_allow_html=True)
        st.subheader("Mitigation Strategy Distribution")
        strat_counts = {s: strategies.count(s) for s in set(strategies)}
        df_strat = pd.DataFrame({"Strategy": list(strat_counts.keys()), "Count": list(strat_counts.values())})
        df_strat = df_strat.sort_values("Count", ascending=False)
        if _PLOTLY:
            palette = ["#E63946","#F4A261","#2A9D8F","#264653","#E9C46A","#6A0572","#E76F51","#457B9D"]
            fig_strat = go.Figure(go.Bar(
                x=df_strat["Strategy"], y=df_strat["Count"],
                marker_color=palette[:len(df_strat)],
                text=df_strat["Count"], textposition="outside"))
            fig_strat.update_layout(xaxis_title="Strategy", yaxis_title="Times Selected",
                margin=dict(l=40,r=20,t=20,b=80), height=320,
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
            st.plotly_chart(fig_strat, use_container_width=True)
        else:
            st.bar_chart(df_strat.set_index("Strategy"))

        st.subheader("Strategy-to-Threat Affinity Map")
        affinity_rows = []
        for threat, strats in THREAT_STRATEGY_AFFINITY.items():
            affinity_rows.append({
                "Threat": THREAT_TYPES.get(threat, {}).get("label", threat),
                "Preferred Strategies": ", ".join(strats),
            })
        st.dataframe(pd.DataFrame(affinity_rows), use_container_width=True, hide_index=True)

        st.subheader("Last Cycle Mitigation Decision")
        last_mit = results[-1]["mitigation"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Strategy", last_mit["selected_strategy"])
        c2.metric("Selection Mode",    last_mit["selection_mode"])
        c3.metric("Threat Profile",    last_mit["threat_profile"])

        st.write("**Action Plan:**")
        for action in last_mit.get("actions", []):
            st.json(action)

        st.write("**Top Q-value Strategies:**")
        for strat, q_val in last_mit.get("top_strategies", []):
            st.write(f"- `{strat}`: {q_val:.4f}")

    # ---- TAB 6: Adaptive Learning ----
    with tab6:
        st.markdown("""
<div style="background:#F3EBF8;border-left:5px solid #7030A0;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#7030A0;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Cross-Layer — Feedback Controller</span><br>
<span style="font-size:15px;font-weight:bold;color:#4B0082;">AdaptiveController + LearningController</span>
<span style="color:#555;font-size:13px;"> — Auto-tunes 4 global parameters every cycle using FP/FN/cascade/mitigation proxy metrics</span>
</div>""", unsafe_allow_html=True)
        st.subheader("Adaptive Parameter Evolution")
        if th_hist:
            df_thresh = pd.DataFrame(th_hist)
            params = ["behavioral_sensitivity", "anomaly_contamination",
                      "cascade_propagation_threshold", "strategy_exploration_rate"]
            df_thresh = df_thresh[[p for p in params if p in df_thresh.columns]]
            df_thresh.index = range(1, len(df_thresh) + 1)
            df_thresh.index.name = "Cycle"
            if _PLOTLY:
                adapt_colors = {"behavioral_sensitivity":"#E63946",
                                "anomaly_contamination":"#2A9D8F",
                                "cascade_propagation_threshold":"#E9C46A",
                                "strategy_exploration_rate":"#6A0572"}
                fig_adapt = go.Figure()
                for col in df_thresh.columns:
                    fig_adapt.add_trace(go.Scatter(
                        x=df_thresh.index, y=df_thresh[col],
                        name=col.replace("_"," ").title(),
                        line=dict(color=adapt_colors.get(col,"#333"), width=2)))
                fig_adapt.update_layout(xaxis_title="Cycle",
                    legend=dict(orientation="h",y=-0.35),
                    margin=dict(l=40,r=20,t=20,b=80), height=320,
                    plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
                st.plotly_chart(fig_adapt, use_container_width=True)
            else:
                st.line_chart(df_thresh)

            st.subheader("Parameter Convergence (Std Dev)")
            conv_data = {p: round(df_thresh[p].std(), 6) for p in params if p in df_thresh.columns}
            st.dataframe(pd.DataFrame({"Parameter": list(conv_data.keys()), "Std Dev": list(conv_data.values())}), use_container_width=True)

            if st.session_state.get("ctrl"):
                ctrl = st.session_state.ctrl
                st.metric("Total Adaptations", ctrl.adaptation_count)
                st.write("**Final Config Snapshot:**")
                st.json(fw.runtime_config.to_dict())

    # ---- TAB 7: Agent Status ----
    with tab7:
        st.markdown("""
<div style="background:#E8F0F8;border-left:5px solid #1F4E79;padding:10px 16px;border-radius:6px;margin-bottom:14px;">
<span style="font-size:13px;color:#1F4E79;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">All Layers — Framework Agents</span><br>
<span style="color:#333;font-size:13px;">Processing latency, call count, and layer assignment for every agent in the pipeline.</span>
</div>""", unsafe_allow_html=True)

        AGENT_META = {
            "data_fusion":        ("Layer 0 — Data Fusion",    "#4472C4", "Unifies network, physical, SCADA, DER into 16-dim feature vector"),
            "behavioral_envelope":("Layer 1 — Perceptual",     "#2E75B6", "Rolling σ-window envelope anomaly detector"),
            "anomaly_detection":  ("Layer 1 — Perceptual",     "#2E75B6", "IsolationForest unsupervised ML outlier detection"),
            "threat_classifier":  ("Layer 2 — Cognitive",      "#375623", "Random Forest 10-class threat classifier (online learning)"),
            "cascade_predictor":  ("Layer 2 — Cognitive",      "#375623", "BFS graph cascade propagation predictor"),
            "mitigation_generator":("Layer 3 — Strategic",     "#843C0C", "ε-greedy Q-Learning mitigation policy planner"),
        }
        st.subheader("Agent Performance Status")
        for agent in fw.get_agents():
            stats    = agent.performance_stats
            avg_time = (stats["total_time_ms"] / stats["total_calls"]) if stats["total_calls"] > 0 else 0
            meta     = AGENT_META.get(agent.agent_id, ("Unknown Layer","#888",""))
            layer_label, layer_color, role_desc = meta
            st.markdown(f"""
<div class="agent-card" style="border-left-color:{layer_color};">
  <div class="agent-layer" style="color:{layer_color};">{layer_label}</div>
  <div class="agent-name">{agent.agent_id.replace("_"," ").title()}</div>
  <div class="agent-role">{role_desc}</div>
  <div class="agent-stats">
    <span class="stat-badge">Calls: {stats["total_calls"]}</span>
    <span class="stat-badge">Avg: {avg_time:.3f} ms</span>
    <span class="stat-badge">Total: {stats["total_time_ms"]:.2f} ms</span>
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()
        # ── Dataset Comparison Panel ─────────────────────────────────────────
        st.subheader("📊 Dataset Comparison")
        dataset_id = st.session_state.get("dataset_id","SYN")
        ds_label   = {"SYN":"Synthetic Generator","DS1":"Dataset 1 (Custom CSV)","DS2":"Dataset 2 (TON_IoT)"}.get(dataset_id,"Current")
        if "comparison_results" not in st.session_state:
            st.session_state["comparison_results"] = {}
        # Store current run metrics
        metrics_now = fw.threat_classifier_agent.get_performance_metrics()
        if "error" not in metrics_now:
            st.session_state["comparison_results"][ds_label] = {
                "Accuracy":           metrics_now["accuracy"],
                "Precision (W.Avg)":  metrics_now["precision_weighted"],
                "Recall (W.Avg)":     metrics_now["recall_weighted"],
                "F1 Score (W.Avg)":   metrics_now["f1_weighted"],
                "Avg Detection (ms)": metrics_now["avg_detection_time_ms"],
                "Total Predictions":  metrics_now["total_predictions"],
            }
        comp = st.session_state["comparison_results"]
        if len(comp) >= 1:
            df_comp = pd.DataFrame(comp).T.reset_index().rename(columns={"index":"Dataset"})
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            if _PLOTLY and len(comp) >= 2:
                st.subheader("Cross-Dataset Performance Comparison")
                metrics_to_plot = ["Accuracy","Precision (W.Avg)","Recall (W.Avg)","F1 Score (W.Avg)"]
                ds_names  = list(comp.keys())
                bar_colors = ["#2E75B6","#E63946","#2A9D8F","#E9C46A"]
                fig_comp = go.Figure()
                for mi, metric in enumerate(metrics_to_plot):
                    vals = [comp[ds].get(metric, 0) for ds in ds_names]
                    fig_comp.add_trace(go.Bar(
                        name=metric, x=ds_names, y=vals,
                        marker_color=bar_colors[mi],
                        text=[f"{v:.3f}" for v in vals], textposition="outside"))
                fig_comp.update_layout(
                    barmode="group", yaxis=dict(range=[0,1.1], title="Score"),
                    xaxis_title="Dataset",
                    legend=dict(orientation="h",y=-0.3),
                    margin=dict(l=40,r=20,t=20,b=80), height=380,
                    plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF")
                st.plotly_chart(fig_comp, use_container_width=True)
                st.caption("Run the simulation on both datasets to populate this chart. Results persist across runs within the same session.")
            elif len(comp) == 1:
                st.info("💡 Run the simulation on a second dataset to see the comparison chart here.")
        else:
            st.info("No metrics yet — run the simulation first.")

        st.divider()
        st.subheader("Monitor Report")
        report = fw.performance_monitor.generate_report()
        if report:
            st.json(report)

else:
    st.info("👈 Configure parameters in the sidebar and click **▶ Run Simulation** to start.")

    st.subheader("🏗️ Enhanced Framework Architecture")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### Layer 0: Data Fusion")
        st.markdown("""
- **DataFusionAgent** — Unifies 4 data domains  
- Network traffic features  
- Physical sensors (V, I, f, T)  
- SCADA telemetry & commands  
- DER setpoints & outputs
        """)
    with col2:
        st.markdown("### Layer 1: Perceptual")
        st.markdown("""
- **BehavioralEnvelopeAgent** — Rolling σ detection  
- **AnomalyDetectionAgent** — IsolationForest ML
        """)
    with col3:
        st.markdown("### Layer 2: Cognitive")
        st.markdown("""
- **ThreatClassifierAgent** — Random Forest threat type prediction (10 threat classes)  
- **CascadePredictorAgent** — BFS graph cascade simulation
        """)
    with col4:
        st.markdown("### Layer 3: Strategic")
        st.markdown("""
- **MitigationGeneratorAgent** — ε-greedy Q-learning  
- Threat-aware strategy affinity mapping  
- 8 response strategies
        """)

    st.subheader("🎯 Threat Taxonomy (10 classes)")
    df_tax = pd.DataFrame([
        {"Threat ID": k, "Threat Name": v["label"], "Severity": "⭐" * v["severity"]}
        for k, v in THREAT_TYPES.items()
    ])
    st.dataframe(df_tax, use_container_width=True, hide_index=True)

    st.subheader("📊 Performance Metrics")
    st.markdown("""
After running a simulation, the **Model Performance** tab shows:
- **Accuracy, Precision, Recall, F1** (weighted across all threat classes)
- **Confusion Matrix** (true vs predicted threat types)
- **Per-class breakdown** (precision / recall / F1 / support per threat)
- **Detection time distribution** (latency per cycle in ms)
    """)
