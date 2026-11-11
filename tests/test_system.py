"""
Automated Unit and Integration Test Suite for the Autonomous Multi-Agent AI System.
Built using standard Python unittest framework (Zero External Dependencies).

Run via:
    cmd.exe /c run.bat tests/test_system.py
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState
from agents import (
    orchestrator,
    data_agent,
    eda_agent,
    ml_agent,
    viz_agent,
    critic_agent,
    insight_agent
)
from graph import build_graph


class TestMultiAgentSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = "datasets/raw/sample.csv"
        cls.heart_path = "datasets/raw/heart_disease.csv"
        assert os.path.exists(cls.sample_path), "Sample dataset missing!"

    def test_01_orchestrator_planning(self):
        state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict whether tumor is malignant and discover key features",
            "agent_trace": []
        }
        res = orchestrator.plan_tasks(state)
        self.assertIn("data_agent", res["task_plan"])
        self.assertIn("eda_agent", res["task_plan"])
        self.assertIn("ml_agent", res["task_plan"])
        self.assertIn("critic_agent", res["task_plan"])
        self.assertIn("insight_agent", res["task_plan"])
        self.assertEqual(res["current_task_index"], 0)

    def test_02_data_agent_hygiene(self):
        state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict tumor",
            "agent_trace": []
        }
        res = data_agent.run(state)
        schema = res.get("schema_report", {})
        self.assertGreater(schema.get("n_rows", 0), 0)
        self.assertGreater(schema.get("n_cols", 0), 0)
        self.assertIn("data_quality_score", schema)
        self.assertTrue(os.path.exists(res.get("cleaned_dataset_path", "")))

    def test_03_eda_statistical_battery(self):
        state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict whether tumor is malignant",
            "agent_trace": []
        }
        state = data_agent.run(state)
        res = eda_agent.run(state)
        eda = res.get("eda_findings", {})
        self.assertIn("top_correlations", eda)
        self.assertIn("normality_tests", eda)
        hyp_tests = res.get("hypothesis_tests", [])
        self.assertGreater(len(hyp_tests), 0)
        # Check first test structure
        t0 = hyp_tests[0]
        self.assertIn("p_value", t0)
        self.assertIn("effect_size", t0)
        self.assertIn("is_significant", t0)

    def test_04_ml_agent_5fold_cv_and_xai(self):
        state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict tumor diagnosis",
            "agent_trace": []
        }
        state = data_agent.run(state)
        state = eda_agent.run(state)
        res = ml_agent.run(state)
        m_rep = res.get("model_report", {})
        self.assertEqual(m_rep.get("problem_type"), "classification")
        self.assertIn("accuracy", m_rep)
        self.assertIn("cv_accuracy", m_rep)
        self.assertIn("overfit_gap", m_rep)
        top_f = res.get("top_features", [])
        self.assertGreater(len(top_f), 0)
        # Check directional impact
        self.assertIn("impact_direction", top_f[0])

    def test_05_critic_4pillar_audit(self):
        state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict tumor",
            "agent_trace": []
        }
        state = data_agent.run(state)
        state = eda_agent.run(state)
        state = ml_agent.run(state)
        res = critic_agent.run(state)
        self.assertIn("validation_status", res)
        self.assertIn("confidence_score", res)
        self.assertGreaterEqual(res["confidence_score"], 60)
        self.assertLessEqual(res["confidence_score"], 100)
        self.assertGreater(len(res["validation_notes"]), 0)

    def test_06_visualization_artifacts(self):
        state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict tumor",
            "agent_trace": []
        }
        state = data_agent.run(state)
        state = eda_agent.run(state)
        state = ml_agent.run(state)
        res = viz_agent.run(state)
        charts = res.get("chart_paths", [])
        self.assertGreaterEqual(len(charts), 4)
        for p in charts:
            self.assertTrue(os.path.exists(p), f"Chart missing: {p}")

    def test_07_full_langgraph_pipeline_integration(self):
        app = build_graph()
        initial_state: AgentState = {
            "dataset_path": self.sample_path,
            "user_goal": "predict whether tumor is malignant based on measurements",
            "agent_trace": []
        }
        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        self.assertIsNotNone(final_state.get("final_report"))
        self.assertIn("Evidence Confidence Score", final_state["final_report"])
        self.assertTrue(os.path.exists(final_state.get("report_pdf_path", "")))
        self.assertTrue(os.path.exists(final_state.get("report_html_path", "")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
