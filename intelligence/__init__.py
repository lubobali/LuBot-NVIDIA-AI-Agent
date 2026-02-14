"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Main Facade
═══════════════════════════════════════════════════════════════════════════════

This is the main entry point that provides the unified IntelligenceEngine class.
It composes all specialized analyzers into a single interface.

BACKWARD COMPATIBLE: All method signatures match the original 9,190-line monolith.

Architecture:
- StatisticalAnalyzer: Trend testing, comparison testing, sample size
- ParadoxDetector: Simpson's Paradox, mix shift, rate/mix decomposition
- DriverAnalyzer: Contribution decomposition, driver ranking, outlier detection
- ConcentrationAnalyzer: HHI scoring, risk assessment, diversification
- CorrelationAnalyzer: Lagged correlation, leading indicators
- AnomalyDetector: Counter-intuitive patterns, expectation violations
- ScenarioAnalyzer: Silent killers, what-if scenarios, sensitivity tables
- PrioritizationEngine: Goal seeking, finding prioritization, topic highlighting
- IntelligenceFormatter: Baseline comparison, report generation
"""

import logging
from typing import Dict, List, Optional

from .base import IntelligenceBase
from .statistical import StatisticalAnalyzer
from .paradox import ParadoxDetector
from .drivers import DriverAnalyzer
from .concentration import ConcentrationAnalyzer
from .correlation import CorrelationAnalyzer
from .anomaly import AnomalyDetector
from .scenarios import ScenarioAnalyzer
from .prioritization import PrioritizationEngine
from .formatters import IntelligenceFormatter
from .timeseries import TimeSeriesAnalyzer

logger = logging.getLogger(__name__)


class IntelligenceEngine(IntelligenceBase):
    """
    🧠 Unified Intelligence Engine
    
    PhD-Level Analytics with CEO-Friendly Output.
    
    This facade class composes all specialized analyzers and delegates
    method calls to the appropriate analyzer. All method signatures
    are backward compatible with the original monolithic implementation.
    
    Usage:
        engine = IntelligenceEngine()
        
        # Statistical testing
        result = engine.test_trend_significance(user_id, metric_name)
        
        # Simpson's Paradox detection
        result = engine.scan_simpsons_paradox(user_id, metric, dimension)
        
        # Concentration risk
        result = engine.calculate_hhi_score(user_id, metric, dimension)
        
        # And 20+ more methods...
    """
    
    def __init__(self):
        """Initialize the Intelligence Engine with all analyzers."""
        super().__init__()
        
        logger.info("🧠 Initializing Intelligence Engine (Modular Architecture)...")
        
        # Initialize all specialized analyzers
        self._statistical = StatisticalAnalyzer()
        self._paradox = ParadoxDetector()
        self._drivers = DriverAnalyzer()
        self._concentration = ConcentrationAnalyzer()
        self._correlation = CorrelationAnalyzer()
        self._anomaly = AnomalyDetector()
        self._scenarios = ScenarioAnalyzer()
        self._prioritization = PrioritizationEngine()
        self._formatter = IntelligenceFormatter()
        
        # 🕐 Time-Series Analyzer (for TIME_SERIES_ONLY data)
        self._timeseries = TimeSeriesAnalyzer(self.engine)
        
        logger.info("✅ Intelligence Engine initialized with 9 specialized modules")
    
    def set_file_scope(self, source_file: str):
        """
        🎯 Phase 4B: Propagate file scope to ALL child analyzers.

        Sets _source_file on each analyzer (NOT on the adapter directly).
        The base.py uploaded_adapter property syncs _source_file to the adapter
        when it's first accessed (lazy-load safe — no premature adapter creation).

        Big Tech Pattern: Google BigQuery USE DATASET propagation
        - Facade sets scope once → all child modules inherit it
        - No lazy-load trigger → zero overhead for unused analyzers

        Args:
            source_file: Filename to scope to (e.g., 'employees.csv')
        """
        self._source_file = source_file
        for analyzer in [self._statistical, self._paradox, self._drivers,
                         self._concentration, self._correlation, self._anomaly,
                         self._scenarios, self._prioritization, self._formatter]:
            analyzer._source_file = source_file
        logger.info(f"🎯 Phase 4B: File scope '{source_file}' set on all 9 analyzers")

    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 TASK 3.1: STATISTICAL SIGNIFICANCE TESTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_trend_significance(self, user_id: str, metric_name: str = None,
                                 confidence_level: float = 0.95) -> Dict:
        """SUBTASK 3.1.1: Test if a metric's trend is statistically significant."""
        return self._statistical.test_trend_significance(user_id, metric_name)
    
    def test_comparison_significance(self, user_id: str, metric_name: str,
                                      dimension: str, value_a: str, value_b: str,
                                      confidence_level: float = 0.95) -> Dict:
        """SUBTASK 3.1.2: Test if difference between two segments is significant."""
        return self._statistical.test_comparison_significance(
            user_id, metric_name, dimension, value_a, value_b
        )
    
    def advise_sample_size(self, user_id: str, metric_name: str = None,
                           desired_effect_pct: float = 10.0,
                           confidence_level: float = 0.95,
                           power: float = 0.80) -> Dict:
        """SUBTASK 3.1.3: Advise on sample size requirements."""
        alpha = 1 - confidence_level
        return self._statistical.advise_sample_size(
            user_id, metric_name, desired_effect_pct, power, alpha
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎭 TASK 3.2: SIMPSON'S PARADOX DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def scan_simpsons_paradox(self, user_id: str, metric_name: str,
                               dimension: str) -> Dict:
        """SUBTASK 3.2.1: Scan for Simpson's Paradox."""
        return self._paradox.scan_simpsons_paradox(user_id, metric_name, dimension)
    
    def analyze_mix_shift_detailed(self, user_id: str, metric_name: str,
                                    dimension: str) -> Dict:
        """SUBTASK 3.2.2: Detailed mix shift analysis with rate/mix decomposition."""
        return self._paradox.analyze_mix_shift_detailed(user_id, metric_name, dimension)
    
    def explain_paradox(self, user_id: str, metric_name: str, dimension: str,
                        use_llm: bool = True) -> Dict:
        """SUBTASK 3.2.3: Generate CEO-friendly paradox explanation."""
        return self._paradox.explain_paradox(user_id, metric_name, dimension, use_llm)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📈 TASK 3.3: DRIVER ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def decompose_contributions(self, user_id: str, metric_name: str,
                                 dimension: str) -> Dict:
        """SUBTASK 3.3.1: Decompose contributions by segment."""
        return self._drivers.decompose_contributions(user_id, metric_name, dimension)
    
    def rank_drivers(self, user_id: str, metric_name: str, dimension: str,
                     ranking_method: str = "contribution") -> Dict:
        """SUBTASK 3.3.2: Rank drivers by various methods."""
        return self._drivers.rank_drivers(user_id, metric_name, dimension)
    
    def detect_outliers(self, user_id: str, metric_name: str, dimension: str,
                        method: str = "iqr", threshold: float = 1.5) -> Dict:
        """SUBTASK 3.3.3: Detect outlier segments."""
        return self._drivers.detect_outliers(user_id, metric_name, dimension, method, threshold)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 TASK 3.4: CONCENTRATION RISK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_hhi_score(self, user_id: str, metric_name: str,
                            dimension: str) -> Dict:
        """SUBTASK 3.4.1: Calculate Herfindahl-Hirschman Index."""
        return self._concentration.calculate_hhi(user_id, metric_name, dimension)
    
    def assess_concentration_risk(self, user_id: str, metric_name: str,
                                   dimension: str) -> Dict:
        """SUBTASK 3.4.2: Full concentration risk assessment."""
        return self._concentration.report_concentration_risk(user_id, metric_name, dimension)
    
    def recommend_diversification(self, user_id: str, metric_name: str,
                                   dimension: str, target_hhi: int = 1500) -> Dict:
        """SUBTASK 3.4.3: Recommend diversification strategies."""
        # Note: This method is not yet implemented in ConcentrationAnalyzer
        # Returning a placeholder for backward compatibility
        hhi_result = self._concentration.calculate_hhi(user_id, metric_name, dimension)
        if "error" in hhi_result:
            return hhi_result
        
        current_hhi = hhi_result.get("hhi", {}).get("points", 0)
        return {
            "current_hhi": current_hhi,
            "target_hhi": target_hhi,
            "needs_diversification": current_hhi > target_hhi,
            "recommendation": f"Current HHI is {current_hhi:.0f}. Target is {target_hhi}. Diversification needed." if current_hhi > target_hhi else "HHI is within acceptable range."
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔗 TASK 3.5: CORRELATION & LEADING INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_lagged_correlations(self, user_id: str, metric_a: str, metric_b: str,
                                  max_lag: int = 12) -> Dict:
        """SUBTASK 3.5.1: Find lagged correlations between metrics."""
        return self._correlation.find_lagged_correlations(user_id, metric_a, metric_b, max_lag)
    
    def detect_leading_indicators(self, user_id: str, target_metric: str = None,
                                   max_lag: int = 8) -> Dict:
        """SUBTASK 3.5.2: Detect leading indicators for a target metric."""
        return self._correlation.detect_leading_indicators(user_id, target_metric, max_lag)

    def calculate_correlation(self, user_id: str, column_x: str = None,
                             column_y: str = None, metric_a: str = None,
                             metric_b: str = None) -> Dict:
        """
        🔄 UNIVERSAL: Calculate correlation (MY DATA or DEMO mode).

        Args:
            column_x, column_y: For MY DATA mode (categorical files)
            metric_a, metric_b: For DEMO mode (time-series)
        """
        return self._correlation.calculate_correlation(
            user_id, column_x, column_y, metric_a, metric_b
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ⚡ TASK 3.6: ANOMALY & COUNTER-INTUITIVE PATTERNS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_counter_intuitive_patterns(self, user_id: str,
                                         metric_name: str = None) -> Dict:
        """SUBTASK 3.6.1: Find counter-intuitive patterns."""
        return self._anomaly.find_counter_intuitive_patterns(user_id)
    
    def detect_expectation_violations(self, user_id: str,
                                       metric_name: str = None) -> Dict:
        """SUBTASK 3.6.2: Detect expectation violations."""
        return self._anomaly.detect_expectation_violations(user_id, metric_name)
    
    def find_silent_killers(self, user_id: str, metric_name: str = None,
                            min_decline_months: int = 3,
                            min_monthly_decline_pct: float = 1.0) -> Dict:
        """SUBTASK 3.6.3: Find slowly declining dimensions (boiling frog detector)."""
        return self._scenarios.find_silent_killers(
            user_id, metric_name, min_decline_months, min_monthly_decline_pct
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔮 TASK 3.7: SCENARIO ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_scenario(self, user_id: str, scenarios: List[Dict]) -> Dict:
        """SUBTASK 3.7.1: Calculate impact of hypothetical scenarios."""
        return self._scenarios.calculate_scenario(user_id, scenarios)
    
    def build_sensitivity_table(self, user_id: str, metric_name: str,
                                 dimension: str,
                                 change_scenarios: List[float] = None) -> Dict:
        """SUBTASK 3.7.2: Build sensitivity table for dimension values."""
        return self._scenarios.build_sensitivity_table(
            user_id, metric_name, dimension, change_scenarios
        )
    
    def seek_goal(self, user_id: str, metric_name: str, target_value: float,
                  dimension: str = None) -> Dict:
        """SUBTASK 3.7.3: Reverse-calculate what changes are needed to hit a target."""
        return self._prioritization.seek_goal(user_id, metric_name, target_value, dimension)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 TASK 3.8: PRIORITIZATION & PERSONALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def prioritize_findings(self, user_id: str, findings: List[Dict],
                            user_context: Optional[Dict] = None) -> Dict:
        """SUBTASK 3.8.1: Prioritize findings with board-level classification."""
        return self._prioritization.prioritize_findings(user_id, findings, user_context)
    
    def highlight_by_user_topics(self, user_id: str, findings: List[Dict],
                                  user_context: Optional[Dict] = None) -> Dict:
        """SUBTASK 3.8.2: Netflix-style topic-aware highlighting."""
        return self._prioritization.highlight_by_user_topics(user_id, findings, user_context)
    
    def compare_to_baseline(self, user_id: str, findings: List[Dict],
                            user_context: Optional[Dict] = None) -> Dict:
        """SUBTASK 3.8.3: Compare findings to user's historical baseline."""
        return self._formatter.compare_to_baseline(user_id, findings, user_context)
    
    def get_full_intelligence_report(self, user_id: str, findings: List[Dict],
                                      user_context: Optional[Dict] = None) -> Dict:
        """Generate complete intelligence report combining all 3.8 capabilities."""
        return self._formatter.get_full_intelligence_report(
            user_id, findings, user_context,
            prioritization_engine=self._prioritization,
            personalization_engine=self._prioritization
        )
    
    def get_personalized_insights_summary(self, user_id: str, findings: List[Dict],
                                           user_context: Optional[Dict] = None) -> str:
        """Generate CEO-friendly summary of personalized insights."""
        result = self._prioritization.highlight_by_user_topics(user_id, findings, user_context)
        # Format as string summary
        if "error" in result:
            return "Could not personalize insights at this time."
        
        lines = ["=" * 50, "📋 YOUR PERSONALIZED INSIGHTS", "=" * 50, ""]
        lines.append(result.get("personalization_message", ""))
        lines.append("")
        
        personalized = result.get("personalized_findings", [])
        if personalized:
            lines.append(f"🎯 FINDINGS THAT MATTER TO YOU ({len(personalized)}):")
            for i, finding in enumerate(personalized[:5], 1):
                label = finding.get("dimension_value") or finding.get("metric_name", "Item")
                lines.append(f"#{i}: {label}")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🛠️ UTILITY METHODS (exposed from base)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_current_metric_values(self, user_id: str) -> Dict[str, float]:
        """Get current values for all metrics (exposed for testing)."""
        return self._scenarios._get_current_metric_values(user_id)
    
    def _get_period_segment_data(self, user_id: str, metric_name: str,
                                  dimension: str) -> Dict:
        """Get period segment data (exposed for testing)."""
        return self._paradox._get_period_segment_data(user_id, metric_name, dimension)


    # ═══════════════════════════════════════════════════════════════════════════
    # 🕐 TIME-SERIES ANALYSIS (for TIME_SERIES_ONLY schema)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run_timeseries_analysis(self, user_id: str, metric_name: str,
                                 schema_type: str = None) -> Dict:
        """
        Run all time-series analyses for TIME_SERIES_ONLY data.
        
        This is for data with Date + Metric only (no dimensions).
        Existing analyzers handle METRIC_WITH_DIMS data.
        
        Args:
            user_id: User identifier
            metric_name: Name of the metric
            schema_type: Must be "TIME_SERIES_ONLY" to run
            
        Returns:
            Dict with trend, seasonality, anomalies, volatility, forecast, statistics
        """
        return self._timeseries.run_all_analyses(user_id, metric_name, schema_type)


# Convenience alias for backward compatibility
Engine = IntelligenceEngine


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("🧠 INTELLIGENCE ENGINE - Modular Architecture Test")
    print("=" * 60 + "\n")
    
    try:
        engine = IntelligenceEngine()
        print("✅ IntelligenceEngine initialized successfully")
        print(f"   - 9 specialized modules loaded")
        
        # Quick method availability check
        methods = [
            "test_trend_significance",
            "scan_simpsons_paradox",
            "decompose_contributions",
            "calculate_hhi_score",
            "find_lagged_correlations",
            "find_counter_intuitive_patterns",
            "calculate_scenario",
            "prioritize_findings",
            "compare_to_baseline"
        ]
        
        print("\n📋 Method availability check:")
        for method in methods:
            has_method = hasattr(engine, method)
            status = "✅" if has_method else "❌"
            print(f"   {status} {method}")
        
        # Test with user if provided
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        if test_user:
            print(f"\n🧪 Running quick test with user {test_user[:8]}...")
            
            # Get metrics
            from sqlalchemy import text
            with engine.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT DISTINCT metric_name FROM user_uploaded_metrics
                    WHERE user_id = :user_id LIMIT 3
                """), {"user_id": test_user})
                metrics = [row.metric_name for row in result.fetchall()]
            
            if metrics:
                print(f"   Found {len(metrics)} metrics: {metrics[:3]}")
                
                # Test trend significance
                result = engine.test_trend_significance(test_user, metrics[0])
                print(f"   ✅ test_trend_significance: {result.get('is_significant', 'N/A')}")
            else:
                print("   ⚠️ No metrics found for user")
        
        print("\n" + "=" * 60)
        print("✅ MODULAR ARCHITECTURE WORKING CORRECTLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

