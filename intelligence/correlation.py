"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Correlation Analysis Module
═══════════════════════════════════════════════════════════════════════════════

Task 3.5: Correlation Analysis
- 3.5.1: Lagged Correlation Finder (Quant Analysis)
- 3.5.2: Leading Indicator Detector (Auto-Discovery)

Big Tech Pattern: Hedge fund quant-style time series analysis.
"""

import math
import logging
from typing import Dict, List, Optional

from sqlalchemy import text

from .base import IntelligenceBase

logger = logging.getLogger(__name__)


class CorrelationAnalyzer(IntelligenceBase):
    """
    ⏱️ Correlation & Leading Indicator Analyzer
    
    Hedge fund quant-style analysis:
    - "Does marketing spend predict sales 2 weeks later?"
    - Auto-discovers leading indicators across all metrics
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⏱️ SUBTASK 3.5.1: LAGGED CORRELATION FINDER (Quant Analysis)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_lagged_correlations(self, user_id: str, metric_a: str, metric_b: str,
                                  max_lag: int = 12) -> Dict:
        """
        SUBTASK 3.5.1: Find correlations between metrics at different time lags.
        
        Hedge Fund Quant Analysis:
        - Tests correlations at lag 0, 1, 2, ... N periods
        - Identifies optimal lag with highest correlation
        """
        try:
            series_a = self._get_time_series_with_dates(user_id, metric_a)
            series_b = self._get_time_series_with_dates(user_id, metric_b)
            
            if not series_a or not series_b:
                return {"error": "Could not retrieve time series data"}
            
            aligned_a, aligned_b = self._align_time_series(series_a, series_b)
            
            if len(aligned_a) < max_lag + 5:
                return {
                    "error": f"Need at least {max_lag + 5} aligned data points, have {len(aligned_a)}"
                }
            
            correlations_by_lag = []
            
            for lag in range(-max_lag, max_lag + 1):
                corr = self._calculate_lagged_correlation(aligned_a, aligned_b, lag)
                
                if corr is not None:
                    correlations_by_lag.append({
                        "lag": lag,
                        "correlation": round(corr, 4),
                        "abs_correlation": round(abs(corr), 4),
                        "strength": self._classify_correlation_strength(corr),
                        "interpretation": self._interpret_lag(lag, corr, metric_a, metric_b)
                    })
            
            if not correlations_by_lag:
                return {"error": "Could not calculate any correlations"}
            
            optimal = max(correlations_by_lag, key=lambda x: x["abs_correlation"])
            lead_lag = self._determine_lead_lag(optimal, metric_a, metric_b)
            lag_0 = next((c for c in correlations_by_lag if c["lag"] == 0), None)
            
            improvement = None
            if lag_0 and optimal["lag"] != 0:
                improvement = optimal["abs_correlation"] - lag_0["abs_correlation"]
            
            n_samples = len(aligned_a) - abs(optimal["lag"])
            is_significant = self._is_correlation_significant(optimal["correlation"], n_samples)
            
            interpretation = self._interpret_lagged_correlation(
                optimal, lag_0, lead_lag, metric_a, metric_b, is_significant, improvement
            )
            
            result = {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "data_points": len(aligned_a),
                "lags_tested": len(correlations_by_lag),
                "correlations_by_lag": correlations_by_lag,
                "optimal_lag": {
                    "lag": optimal["lag"],
                    "correlation": optimal["correlation"],
                    "strength": optimal["strength"],
                    "is_significant": is_significant,
                    "p_value_approx": self._correlation_p_value(optimal["correlation"], n_samples)
                },
                "lag_0_correlation": lag_0["correlation"] if lag_0 else None,
                "improvement_from_lag_0": round(improvement, 4) if improvement else None,
                "lead_lag_relationship": lead_lag,
                "interpretation": interpretation,
                "actionable_insight": self._generate_lag_insight(optimal, lead_lag, metric_a, metric_b)
            }
            
            logger.info(f"⏱️ Lagged correlation for user {user_id[:8]}...: {metric_a} vs {metric_b}, optimal lag={optimal['lag']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Lagged correlation failed: {e}")
            return {"error": str(e)}
    
    def _align_time_series(self, series_a: List[Dict], series_b: List[Dict]) -> tuple:
        """Align two time series by date."""
        dict_a = {item["date"]: item["value"] for item in series_a}
        dict_b = {item["date"]: item["value"] for item in series_b}
        
        common_dates = sorted(set(dict_a.keys()) & set(dict_b.keys()))
        
        aligned_a = [dict_a[date] for date in common_dates]
        aligned_b = [dict_b[date] for date in common_dates]
        
        return aligned_a, aligned_b
    
    def _calculate_lagged_correlation(self, series_a: List[float], series_b: List[float],
                                       lag: int) -> Optional[float]:
        """Calculate Pearson correlation with specified lag."""
        try:
            n = len(series_a)
            
            if lag >= 0:
                if lag == 0:
                    a = series_a
                    b = series_b
                else:
                    a = series_a[:-lag] if lag < n else []
                    b = series_b[lag:] if lag < n else []
            else:
                abs_lag = abs(lag)
                a = series_a[abs_lag:] if abs_lag < n else []
                b = series_b[:-abs_lag] if abs_lag < n else []
            
            if len(a) < 3 or len(b) < 3 or len(a) != len(b):
                return None
            
            n_pairs = len(a)
            mean_a = sum(a) / n_pairs
            mean_b = sum(b) / n_pairs
            
            numerator = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n_pairs))
            
            sum_sq_a = sum((x - mean_a) ** 2 for x in a)
            sum_sq_b = sum((x - mean_b) ** 2 for x in b)
            
            denominator = math.sqrt(sum_sq_a * sum_sq_b)
            
            if denominator == 0:
                return None
            
            return numerator / denominator
            
        except Exception:
            return None
    
    def _classify_correlation_strength(self, corr: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(corr)
        
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _interpret_lag(self, lag: int, corr: float, metric_a: str, metric_b: str) -> str:
        """Interpret a single lag correlation."""
        direction = "positive" if corr > 0 else "negative"
        
        if lag == 0:
            return f"Concurrent {direction} correlation"
        elif lag > 0:
            return f"{metric_a} leads {metric_b} by {lag} periods ({direction})"
        else:
            return f"{metric_b} leads {metric_a} by {abs(lag)} periods ({direction})"
    
    def _determine_lead_lag(self, optimal: Dict, metric_a: str, metric_b: str) -> Dict:
        """Determine which metric leads."""
        lag = optimal["lag"]
        
        if lag == 0:
            return {
                "relationship": "concurrent",
                "leading_metric": None,
                "lagging_metric": None,
                "lag_periods": 0,
                "description": f"{metric_a} and {metric_b} move together with no lead/lag"
            }
        elif lag > 0:
            return {
                "relationship": "leading",
                "leading_metric": metric_a,
                "lagging_metric": metric_b,
                "lag_periods": lag,
                "description": f"{metric_a} leads {metric_b} by {lag} periods"
            }
        else:
            return {
                "relationship": "lagging",
                "leading_metric": metric_b,
                "lagging_metric": metric_a,
                "lag_periods": abs(lag),
                "description": f"{metric_b} leads {metric_a} by {abs(lag)} periods"
            }
    
    def _is_correlation_significant(self, corr: float, n: int, alpha: float = 0.05) -> bool:
        """Test if correlation is statistically significant."""
        if n < 4:
            return False
        
        p_value = self._correlation_p_value(corr, n)
        
        return p_value < alpha
    
    def _correlation_p_value(self, corr: float, n: int) -> float:
        """Calculate approximate p-value for correlation."""
        if n < 4 or abs(corr) >= 1:
            return 1.0
        
        t_stat = abs(corr) * math.sqrt((n - 2) / (1 - corr ** 2))
        
        if n > 30:
            p_value = 2 * (1 - 0.5 * (1 + math.erf(t_stat / math.sqrt(2))))
        else:
            p_value = 2 * math.exp(-0.717 * t_stat - 0.416 * t_stat ** 2)
        
        return min(1.0, max(0.0, p_value))
    
    def _interpret_lagged_correlation(self, optimal: Dict, lag_0: Optional[Dict],
                                       lead_lag: Dict, metric_a: str, metric_b: str,
                                       is_significant: bool, improvement: Optional[float]) -> str:
        """Generate comprehensive interpretation."""
        interpretation = f"⏱️ LAGGED CORRELATION ANALYSIS\n\n"
        interpretation += f"Metrics: {metric_a} vs {metric_b}\n\n"
        
        if optimal["lag"] == 0:
            interpretation += (
                f"📊 FINDING: No lead/lag relationship detected.\n"
                f"These metrics move together simultaneously (r={optimal['correlation']:.3f}).\n\n"
            )
        else:
            interpretation += (
                f"📊 FINDING: {lead_lag['description']}\n"
                f"Optimal correlation r={optimal['correlation']:.3f} at lag {optimal['lag']}\n"
            )
            
            if improvement and improvement > 0.05:
                interpretation += (
                    f"This is {improvement:.3f} stronger than concurrent correlation.\n\n"
                )
        
        if is_significant:
            interpretation += f"✅ This relationship is statistically significant.\n"
        else:
            interpretation += f"⚠️ This relationship is NOT statistically significant.\n"
        
        interpretation += f"Correlation strength: {optimal['strength'].upper()}\n\n"
        
        if optimal["lag"] > 0 and optimal["correlation"] > 0.4:
            interpretation += (
                f"💡 INSIGHT: {metric_a} may be a LEADING INDICATOR for {metric_b}.\n"
                f"Changes in {metric_a} appear to predict {metric_b} {optimal['lag']} periods later."
            )
        elif optimal["lag"] < 0 and optimal["correlation"] > 0.4:
            interpretation += (
                f"💡 INSIGHT: {metric_b} may be a LEADING INDICATOR for {metric_a}.\n"
                f"Changes in {metric_b} appear to predict {metric_a} {abs(optimal['lag'])} periods later."
            )
        
        return interpretation
    
    def _generate_lag_insight(self, optimal: Dict, lead_lag: Dict,
                               metric_a: str, metric_b: str) -> str:
        """Generate actionable insight from lagged correlation."""
        if optimal["lag"] == 0:
            return f"Monitor {metric_a} and {metric_b} together — they move in sync."
        
        leading = lead_lag.get("leading_metric")
        lagging = lead_lag.get("lagging_metric")
        lag_periods = lead_lag.get("lag_periods")
        
        if abs(optimal["correlation"]) > 0.6:
            return f"🎯 ACTION: Use {leading} as a {lag_periods}-period early warning for {lagging}."
        elif abs(optimal["correlation"]) > 0.4:
            return f"📊 MONITOR: {leading} shows moderate predictive value for {lagging} ({lag_periods} periods)."
        else:
            return f"ℹ️ Weak lead/lag relationship — not reliable for prediction."
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔮 SUBTASK 3.5.2: LEADING INDICATOR DETECTOR (Auto-Discovery)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_leading_indicators(self, user_id: str, target_metric: str = None,
                                   max_lag: int = 8, min_correlation: float = 0.4) -> Dict:
        """
        SUBTASK 3.5.2: Automatically discover leading indicators.
        
        Scans ALL metric pairs to find which metrics predict others.
        """
        try:
            all_metrics = self._get_user_metrics(user_id)
            
            if len(all_metrics) < 2:
                return {"error": f"Need at least 2 metrics, found {len(all_metrics)}"}
            
            if target_metric:
                if target_metric not in all_metrics:
                    return {"error": f"Target metric '{target_metric}' not found"}
                target_metrics = [target_metric]
            else:
                target_metrics = all_metrics
            
            all_relationships = []
            prediction_network = {}
            
            for target in target_metrics:
                predictors_for_target = []
                
                for candidate in all_metrics:
                    if candidate == target:
                        continue
                    
                    result = self.find_lagged_correlations(
                        user_id, candidate, target, max_lag=max_lag
                    )
                    
                    if "error" in result:
                        continue
                    
                    optimal = result.get("optimal_lag", {})
                    correlation = optimal.get("correlation", 0)
                    lag = optimal.get("lag", 0)
                    
                    if abs(correlation) >= min_correlation and lag > 0:
                        relationship = {
                            "leading_metric": candidate,
                            "lagging_metric": target,
                            "lag_periods": lag,
                            "correlation": round(correlation, 4),
                            "abs_correlation": round(abs(correlation), 4),
                            "strength": optimal.get("strength"),
                            "is_significant": optimal.get("is_significant"),
                            "predictive_power": self._calculate_predictive_power(correlation, lag),
                            "description": f"{candidate} predicts {target} {lag} periods ahead (r={correlation:.3f})"
                        }
                        
                        all_relationships.append(relationship)
                        predictors_for_target.append(relationship)
                
                if predictors_for_target:
                    prediction_network[target] = sorted(
                        predictors_for_target,
                        key=lambda x: x["predictive_power"],
                        reverse=True
                    )
            
            all_relationships.sort(key=lambda x: x["predictive_power"], reverse=True)
            
            top_predictors = self._identify_top_predictors(all_relationships)
            recommendations = self._generate_predictor_recommendations(
                all_relationships, top_predictors
            )
            
            summary = {
                "metrics_analyzed": len(all_metrics),
                "pairs_tested": len(all_metrics) * (len(all_metrics) - 1),
                "relationships_found": len(all_relationships),
                "strong_predictors": len([r for r in all_relationships if r["abs_correlation"] >= 0.6]),
                "moderate_predictors": len([r for r in all_relationships if 0.4 <= r["abs_correlation"] < 0.6])
            }
            
            result = {
                "target_metric": target_metric,
                "leading_indicators": all_relationships[:20],
                "prediction_network": prediction_network,
                "top_predictors": top_predictors,
                "recommendations": recommendations,
                "summary": summary
            }
            
            logger.info(f"🔮 Leading indicators for user {user_id[:8]}...: found {len(all_relationships)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"❌ Leading indicator detection failed: {e}")
            return {"error": str(e)}
    
    def _calculate_predictive_power(self, correlation: float, lag: int) -> float:
        """Calculate predictive power score."""
        abs_corr = abs(correlation)
        
        if lag <= 0:
            lag_factor = 0
        elif lag <= 3:
            lag_factor = lag * 0.1
        elif lag <= 7:
            lag_factor = 0.3 + (lag - 3) * 0.05
        else:
            lag_factor = 0.5 - (lag - 7) * 0.05
        
        power = abs_corr + max(0, lag_factor)
        
        return round(power, 4)
    
    def _identify_top_predictors(self, relationships: List[Dict]) -> List[Dict]:
        """Identify the best leading indicators overall."""
        if not relationships:
            return []
        
        by_leader = {}
        for rel in relationships:
            leader = rel["leading_metric"]
            if leader not in by_leader:
                by_leader[leader] = []
            by_leader[leader].append(rel)
        
        leader_scores = []
        for leader, rels in by_leader.items():
            avg_power = sum(r["predictive_power"] for r in rels) / len(rels)
            max_power = max(r["predictive_power"] for r in rels)
            targets_predicted = len(rels)
            
            leader_scores.append({
                "metric": leader,
                "targets_predicted": targets_predicted,
                "avg_predictive_power": round(avg_power, 3),
                "max_predictive_power": round(max_power, 3),
                "overall_score": round(avg_power * (1 + 0.1 * targets_predicted), 3),
                "best_prediction": max(rels, key=lambda x: x["predictive_power"])["description"]
            })
        
        leader_scores.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return leader_scores[:5]
    
    def _generate_predictor_recommendations(self, relationships: List[Dict],
                                             top_predictors: List[Dict]) -> List[Dict]:
        """Generate recommendations for using leading indicators."""
        recommendations = []
        
        if not relationships:
            recommendations.append({
                "priority": "ℹ️ INFO",
                "action": "Collect more data or metrics",
                "rationale": "No strong leading indicators found in current data"
            })
            return recommendations
        
        if top_predictors:
            best = top_predictors[0]
            recommendations.append({
                "priority": "🎯 HIGH",
                "action": f"Monitor {best['metric']} as your primary early warning",
                "rationale": f"Predicts {best['targets_predicted']} metrics with avg power {best['avg_predictive_power']:.2f}",
                "specific": best["best_prediction"]
            })
        
        strong = [r for r in relationships if r["abs_correlation"] >= 0.6]
        if strong:
            recommendations.append({
                "priority": "🎯 HIGH",
                "action": "Build leading indicator dashboard",
                "rationale": f"{len(strong)} strong predictive relationships found",
                "specific": f"Top: {strong[0]['description']}"
            })
        
        revenue_predictors = [r for r in relationships if r["lagging_metric"] in ["revenue", "profit", "sales"]]
        if revenue_predictors:
            best_rev = max(revenue_predictors, key=lambda x: x["predictive_power"])
            recommendations.append({
                "priority": "💰 REVENUE",
                "action": f"Use {best_rev['leading_metric']} to forecast {best_rev['lagging_metric']}",
                "rationale": f"{best_rev['lag_periods']} period advance warning",
                "specific": best_rev["description"]
            })
        
        multi_lag = {}
        for r in relationships:
            target = r["lagging_metric"]
            if target not in multi_lag:
                multi_lag[target] = []
            multi_lag[target].append(r)
        
        for target, rels in multi_lag.items():
            if len(rels) >= 3:
                recommendations.append({
                    "priority": "📊 ENSEMBLE",
                    "action": f"Combine {len(rels)} indicators to predict {target}",
                    "rationale": "Multiple leading indicators improve forecast accuracy",
                    "specific": ", ".join([r["leading_metric"] for r in rels[:3]])
                })
                break
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 UNIVERSAL METHOD: Simple Correlation for MY DATA Mode
    # ═══════════════════════════════════════════════════════════════════════════

    def calculate_correlation(self, user_id: str, column_x: str = None,
                             column_y: str = None, metric_a: str = None,
                             metric_b: str = None) -> Dict:
        """
        🔄 UNIVERSAL correlation calculator - works for DEMO and MY DATA modes.

        BIG TECH PATTERN: Data source detection → route to correct method.

        Args:
            user_id: User identifier
            column_x, column_y: Column names for MY DATA mode (categorical files)
            metric_a, metric_b: Metric names for DEMO mode (time-series)

        Returns:
            Correlation result with coefficient and interpretation

        Example (MY DATA):
            calculate_correlation(user_id, column_x="Age", column_y="Salary")
            → {"correlation": 0.58, "strength": "moderate positive"}

        Example (DEMO):
            calculate_correlation(user_id, metric_a="clicks", metric_b="revenue")
            → Uses time-series correlation
        """
        try:
            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Detect data source (Big Tech adapter pattern)
            # ═══════════════════════════════════════════════════════════════════
            data_source = self._detect_data_source(user_id)
            logger.info(f"🔍 Data source detected: {data_source}")

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Route to correct method based on data source
            # ═══════════════════════════════════════════════════════════════════

            if data_source == "rows_table":
                # MY DATA MODE - Categorical/uploaded data
                logger.info("📊 Using MY DATA correlation (uploaded file)")

                # Get DataFrame using uploaded adapter (do this FIRST)
                df = self.uploaded_adapter.get_dataframe(user_id)

                if df.empty:
                    return {"error": "No uploaded data found"}

                if not column_x or not column_y:
                    # Auto-detect: Use first two numeric columns (Big Tech: pandas standard)
                    import pandas as pd
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) < 2:
                        return {
                            "error": "Need at least 2 numeric columns for correlation",
                            "numeric_columns_found": len(numeric_cols),
                            "available_columns": list(df.columns)
                        }
                    column_x = numeric_cols[0]
                    column_y = numeric_cols[1]
                    logger.info(f"📊 Auto-detected columns: {column_x} vs {column_y}")

                if column_x not in df.columns or column_y not in df.columns:
                    return {
                        "error": f"Columns not found: {column_x}, {column_y}",
                        "available_columns": list(df.columns)
                    }

                # Ensure both columns are numeric
                import pandas as pd
                try:
                    x = pd.to_numeric(df[column_x], errors='coerce')
                    y = pd.to_numeric(df[column_y], errors='coerce')

                    # Remove NaN values
                    valid_mask = x.notna() & y.notna()
                    x_clean = x[valid_mask]
                    y_clean = y[valid_mask]

                    if len(x_clean) < 3:
                        return {
                            "error": "Need at least 3 valid data points for correlation",
                            "valid_points": len(x_clean)
                        }

                except (ValueError, TypeError) as e:
                    return {"error": f"Columns must be numeric: {str(e)}"}

                # ═══════════════════════════════════════════════════════════════
                # STEP 3: Calculate Pearson correlation (REUSE pandas math!)
                # ═══════════════════════════════════════════════════════════════
                correlation = x_clean.corr(y_clean)

                # Statistical significance (t-test)
                n = len(x_clean)
                if n > 2:
                    import math
                    t_stat = correlation * math.sqrt((n - 2) / (1 - correlation**2)) if abs(correlation) < 1 else float('inf')
                    # Approximate p-value (two-tailed)
                    from scipy import stats
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_value = 1.0

                # Interpret strength
                strength = self._interpret_correlation_strength(correlation)

                logger.info(f"✅ Correlation calculated: {correlation:.3f} ({strength})")

                return {
                    "column_x": column_x,
                    "column_y": column_y,
                    "correlation": round(correlation, 3),
                    "strength": strength,
                    "sample_size": n,
                    "p_value": round(p_value, 4),
                    "is_significant": p_value < 0.05,
                    "interpretation": self._interpret_correlation_result(
                        column_x, column_y, correlation, p_value
                    )
                }

            else:
                # DEMO MODE / Time-series data
                logger.info("📊 Using DEMO mode correlation (time-series)")

                # For time-series, use existing lagged correlation method
                if metric_a and metric_b:
                    return self.find_lagged_correlations(
                        user_id, metric_a, metric_b, max_lag=12
                    )
                else:
                    return {
                        "error": "For DEMO mode, provide metric_a and metric_b parameters",
                        "example": "calculate_correlation(user_id, metric_a='clicks', metric_b='revenue')"
                    }

        except Exception as e:
            logger.error(f"❌ Correlation calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _interpret_correlation_strength(self, corr: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "no"

        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        return f"{strength} {direction}" if direction != "no" else "no correlation"

    def _interpret_correlation_result(self, col_x: str, col_y: str,
                                     corr: float, p_value: float) -> str:
        """Generate CEO-friendly interpretation."""
        abs_corr = abs(corr)
        direction = "increases" if corr > 0 else "decreases"

        if p_value >= 0.05:
            return f"No statistically significant relationship found between {col_x} and {col_y}."

        if abs_corr >= 0.7:
            return f"Strong relationship: As {col_x} {direction}, {col_y} tends to {direction} proportionally."
        elif abs_corr >= 0.4:
            return f"Moderate relationship: {col_x} and {col_y} tend to move together, but other factors also play a role."
        elif abs_corr >= 0.2:
            return f"Weak relationship: {col_x} and {col_y} show some connection, but it's not reliable for predictions."
        else:
            return f"Little to no relationship between {col_x} and {col_y}."


if __name__ == "__main__":
    import sys
    
    print("⏱️ Testing CorrelationAnalyzer...")
    
    try:
        analyzer = CorrelationAnalyzer()
        print("✅ CorrelationAnalyzer initialized")
        
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        
        if test_user:
            metrics = analyzer._get_user_metrics(test_user)
            if len(metrics) >= 2:
                result = analyzer.find_lagged_correlations(test_user, metrics[0], metrics[1])
                print(f"✅ Lagged correlation: optimal lag={result.get('optimal_lag', {}).get('lag', 'N/A')}")
                
                indicators = analyzer.detect_leading_indicators(test_user)
                print(f"✅ Leading indicators: {len(indicators.get('leading_indicators', []))} found")
        
        print("\n✅ correlation.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

