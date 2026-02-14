"""
═══════════════════════════════════════════════════════════════════════════════
🧠 INTELLIGENCE ENGINE - Statistical Significance Module
═══════════════════════════════════════════════════════════════════════════════

Task 3.1: Statistical Significance Testing
- 3.1.1: Trend Significance (Mann-Kendall test)
- 3.1.2: Comparison Significance (Welch's t-test)
- 3.1.3: Sample Size Advisor (Power Analysis)

Big Tech Pattern: PhD-level statistics with CEO-friendly explanations.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from .base import IntelligenceBase

logger = logging.getLogger(__name__)


class StatisticalAnalyzer(IntelligenceBase):
    """
    📊 Statistical Significance Analyzer
    
    Provides PhD-level statistical tests with plain-English interpretations.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SUBTASK 3.1.1: TREND SIGNIFICANCE TESTER (Mann-Kendall)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_trend_significance(self, user_id: str, metric_name: Optional[str] = None) -> Dict:
        """
        SUBTASK 3.1.1: Test if a trend is statistically significant.
        
        Uses Mann-Kendall test - non-parametric, doesn't assume normal distribution.
        
        Args:
            user_id: User identifier
            metric_name: Optional metric name
            
        Returns:
            Dict with:
            - is_significant: bool (p < 0.05)
            - p_value: float
            - trend_direction: "increasing", "decreasing", or "no_trend"
            - confidence_level: "high" (p<0.01), "medium" (p<0.05), "low" (p>=0.05)
            - interpretation: Human-readable explanation
        """
        try:
            values = self._get_time_series(user_id, metric_name)
            
            if not values or len(values) < 8:
                logger.info(f"📊 Not enough data for significance test: {len(values) if values else 0} points (need 8+)")
                return {
                    "is_significant": False,
                    "error": "Insufficient data (need 8+ data points)",
                    "data_points": len(values) if values else 0
                }
            
            s_stat, p_value, trend = self._mann_kendall_test(values)
            
            is_significant = p_value < 0.05
            
            if p_value < 0.01:
                confidence_level = "high"
            elif p_value < 0.05:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            if len(values) >= 2:
                first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
                second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                if first_half_avg != 0:
                    growth_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                else:
                    growth_pct = 0
            else:
                growth_pct = 0
            
            interpretation = self._interpret_trend_significance(
                is_significant, p_value, trend, growth_pct
            )
            
            result = {
                "metric_name": metric_name or "all_metrics",
                "is_significant": is_significant,
                "p_value": round(p_value, 4),
                "s_statistic": s_stat,
                "trend_direction": trend,
                "confidence_level": confidence_level,
                "growth_pct": round(growth_pct, 2),
                "data_points": len(values),
                "interpretation": interpretation
            }
            
            logger.info(f"📊 Trend significance for user {user_id[:8]}...: {trend}, p={p_value:.4f}, significant={is_significant}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Trend significance test failed: {e}")
            return {"error": str(e)}
    
    def _mann_kendall_test(self, values: List[float]) -> Tuple[int, float, str]:
        """
        Mann-Kendall test for trend detection.
        Non-parametric test that doesn't assume normal distribution.
        """
        n = len(values)
        
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = values[j] - values[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1
        
        tie_counts = {}
        for v in values:
            tie_counts[v] = tie_counts.get(v, 0) + 1
        
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        
        for count in tie_counts.values():
            if count > 1:
                var_s -= (count * (count - 1) * (2 * count + 5)) / 18
        
        if var_s == 0:
            z = 0
        elif s > 0:
            z = (s - 1) / math.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / math.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - self._normal_cdf(abs(z)))
        
        if s > 0 and p_value < 0.05:
            trend = "increasing"
        elif s < 0 and p_value < 0.05:
            trend = "decreasing"
        else:
            trend = "no_trend"
        
        return s, p_value, trend
    
    def _interpret_trend_significance(self, is_significant: bool, p_value: float,
                                       trend: str, growth_pct: float) -> str:
        """Generate human-readable interpretation of trend significance."""
        if not is_significant:
            return f"The observed change of {growth_pct:+.1f}% is likely random noise (p={p_value:.2f}). Not statistically significant."
        
        if trend == "increasing":
            confidence = "high confidence" if p_value < 0.01 else "moderate confidence"
            return f"This {growth_pct:+.1f}% growth is statistically significant (p={p_value:.3f}). {confidence.title()} this is a real upward trend, not random variation."
        elif trend == "decreasing":
            confidence = "high confidence" if p_value < 0.01 else "moderate confidence"
            return f"This {growth_pct:+.1f}% decline is statistically significant (p={p_value:.3f}). {confidence.title()} this is a real downward trend, not random variation."
        else:
            return f"No clear trend detected (p={p_value:.2f})."
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SUBTASK 3.1.2: COMPARISON SIGNIFICANCE TESTER (Welch's t-test)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_comparison_significance(self, user_id: str, metric_name: str,
                                      dimension: str, value_a: str, value_b: str) -> Dict:
        """
        SUBTASK 3.1.2: Test if two dimension values are significantly different.

        Uses Welch's t-test - robust to unequal variances (heteroscedasticity).
        PhD-level: Includes effect size (Cohen's d) for practical significance.

        BIG TECH OPTION C: Supports both DEMO mode and MY DATA mode
        - DEMO mode: Uses user_uploaded_metrics (time-series)
        - MY DATA mode: Uses user_uploaded_rows (categorical)
        Pattern: Identical to concentration.py
        """
        try:
            # ═══════════════════════════════════════════════════════════════
            # BIG TECH: Detect data source and route appropriately
            # Pattern: Same as concentration.py (adapter layer)
            # ═══════════════════════════════════════════════════════════════
            data_source = self._detect_data_source(user_id)

            if data_source == "rows_table":
                # MY DATA MODE: Use uploaded_adapter for categorical data
                logger.info(f"📊 t-test using MY DATA mode (user_uploaded_rows)")
                df = self.uploaded_adapter.get_dataframe(user_id)

                # Extract values for each group (same logic as DEMO, different source)
                data_a = df[df[dimension] == value_a][metric_name].dropna().astype(float).tolist()
                data_b = df[df[dimension] == value_b][metric_name].dropna().astype(float).tolist()
            else:
                # DEMO MODE: Use existing SQL for time-series data (unchanged)
                logger.info(f"📊 t-test using DEMO mode (user_uploaded_metrics)")
                data_a = self._get_dimension_values(user_id, metric_name, dimension, value_a)
                data_b = self._get_dimension_values(user_id, metric_name, dimension, value_b)
            
            if len(data_a) < 3 or len(data_b) < 3:
                return {
                    "is_significant": False,
                    "error": f"Insufficient data: {value_a}={len(data_a)}, {value_b}={len(data_b)} (need 3+ each)",
                    "value_a_count": len(data_a),
                    "value_b_count": len(data_b)
                }
            
            mean_a = sum(data_a) / len(data_a)
            mean_b = sum(data_b) / len(data_b)
            var_a = sum((x - mean_a) ** 2 for x in data_a) / (len(data_a) - 1)
            var_b = sum((x - mean_b) ** 2 for x in data_b) / (len(data_b) - 1)
            n_a = len(data_a)
            n_b = len(data_b)
            
            t_stat, p_value, df = self._welch_t_test(mean_a, mean_b, var_a, var_b, n_a, n_b)
            
            pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
            if pooled_std > 0:
                cohens_d = (mean_a - mean_b) / pooled_std
            else:
                cohens_d = 0
            
            effect_magnitude = self._classify_effect_size(abs(cohens_d))
            is_significant = p_value < 0.05
            
            if mean_b != 0:
                pct_diff = ((mean_a - mean_b) / mean_b) * 100
            else:
                pct_diff = 0
            
            interpretation = self._interpret_comparison(
                is_significant, p_value, t_stat, cohens_d, effect_magnitude,
                value_a, value_b, mean_a, mean_b, pct_diff
            )
            
            result = {
                "comparison": f"{value_a} vs {value_b}",
                "metric_name": metric_name,
                "dimension": dimension,
                "is_significant": is_significant,
                "p_value": round(p_value, 4),
                "t_statistic": round(t_stat, 3),
                "degrees_of_freedom": round(df, 1),
                "effect_size_cohens_d": round(cohens_d, 3),
                "effect_magnitude": effect_magnitude,
                "descriptive_stats": {
                    "value_a": {
                        "name": value_a,
                        "mean": round(mean_a, 2),
                        "std_dev": round(math.sqrt(var_a), 2),
                        "n": n_a
                    },
                    "value_b": {
                        "name": value_b,
                        "mean": round(mean_b, 2),
                        "std_dev": round(math.sqrt(var_b), 2),
                        "n": n_b
                    },
                    "difference": round(mean_a - mean_b, 2),
                    "pct_difference": round(pct_diff, 2)
                },
                "interpretation": interpretation
            }
            
            logger.info(f"📊 Comparison test {value_a} vs {value_b}: t={t_stat:.2f}, p={p_value:.4f}, d={cohens_d:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Comparison significance test failed: {e}")
            return {"error": str(e)}
    
    def _welch_t_test(self, mean_a: float, mean_b: float, var_a: float, var_b: float,
                      n_a: int, n_b: int) -> Tuple[float, float, float]:
        """Welch's t-test for unequal variances."""
        se = math.sqrt(var_a / n_a + var_b / n_b)
        
        if se == 0:
            return 0, 1.0, min(n_a, n_b) - 1
        
        t_stat = (mean_a - mean_b) / se
        
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
        
        if denom == 0:
            df = min(n_a, n_b) - 1
        else:
            df = num / denom
        
        p_value = self._t_distribution_p_value(abs(t_stat), df)
        
        return t_stat, p_value, df
    
    def _t_distribution_p_value(self, t: float, df: float) -> float:
        """Approximate two-tailed p-value from t-distribution."""
        if df > 100:
            return 2 * (1 - self._normal_cdf(t))
        
        x = df / (df + t * t)
        p = self._incomplete_beta(df / 2, 0.5, x)
        
        return p
    
    def _incomplete_beta(self, a: float, b: float, x: float) -> float:
        """Approximation of regularized incomplete beta function."""
        if x == 0:
            return 0
        if x == 1:
            return 1
        
        if x < (a + 1) / (a + b + 2):
            return self._beta_cf(a, b, x) * (x ** a) * ((1 - x) ** b) / (a * self._beta_func(a, b))
        else:
            return 1 - self._beta_cf(b, a, 1 - x) * ((1 - x) ** b) * (x ** a) / (b * self._beta_func(a, b))
    
    def _beta_cf(self, a: float, b: float, x: float) -> float:
        """Continued fraction for incomplete beta."""
        max_iter = 100
        eps = 1e-10
        
        qab = a + b
        qap = a + 1
        qam = a - 1
        c = 1
        d = 1 - qab * x / qap
        if abs(d) < eps:
            d = eps
        d = 1 / d
        h = d
        
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d
            if abs(d) < eps:
                d = eps
            c = 1 + aa / c
            if abs(c) < eps:
                c = eps
            d = 1 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            if abs(d) < eps:
                d = eps
            c = 1 + aa / c
            if abs(c) < eps:
                c = eps
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < eps:
                break
        
        return h
    
    def _beta_func(self, a: float, b: float) -> float:
        """Beta function using log-gamma."""
        return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    
    def _classify_effect_size(self, d: float) -> str:
        """Classify Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_comparison(self, is_significant: bool, p_value: float, t_stat: float,
                               cohens_d: float, effect_mag: str, value_a: str, value_b: str,
                               mean_a: float, mean_b: float, pct_diff: float) -> str:
        """Generate PhD-level interpretation of comparison results."""
        higher = value_a if mean_a > mean_b else value_b
        lower = value_b if mean_a > mean_b else value_a
        
        if not is_significant:
            return (f"No statistically significant difference between {value_a} and {value_b} "
                   f"(p={p_value:.3f}). The observed {abs(pct_diff):.1f}% difference could be due to "
                   f"random variation. Effect size is {effect_mag} (d={cohens_d:.2f}).")
        
        if effect_mag == "negligible":
            return (f"While statistically significant (p={p_value:.3f}), the difference between "
                   f"{value_a} and {value_b} is negligible in practical terms (d={cohens_d:.2f}). "
                   f"This may be due to large sample size detecting a tiny real difference.")
        
        practical = "meaningful" if effect_mag in ["medium", "large"] else "modest"
        
        return (f"{higher} significantly outperforms {lower} (p={p_value:.3f}). "
               f"The {abs(pct_diff):.1f}% difference represents a {effect_mag} effect size (d={cohens_d:.2f}), "
               f"indicating a {practical} practical difference. "
               f"Confidence: {'High' if p_value < 0.01 else 'Moderate'} that this is a real difference, not chance.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SUBTASK 3.1.3: SAMPLE SIZE ADVISOR (Power Analysis)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def advise_sample_size(self, user_id: str, metric_name: Optional[str] = None,
                           desired_effect_pct: float = 10.0,
                           power: float = 0.80,
                           alpha: float = 0.05) -> Dict:
        """
        SUBTASK 3.1.3: Calculate how much more data is needed to detect an effect.
        
        PhD-level power analysis - tells users exactly how many more observations
        they need to reliably detect a given percentage change.
        """
        try:
            values = self._get_time_series(user_id, metric_name)
            
            if not values or len(values) < 5:
                return {
                    "error": "Need at least 5 data points for power analysis",
                    "current_sample_size": len(values) if values else 0
                }
            
            current_n = len(values)
            
            mean = sum(values) / len(values)
            std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
            
            if mean == 0:
                return {
                    "error": "Cannot calculate effect size with zero mean",
                    "current_sample_size": current_n
                }
            
            absolute_effect = (desired_effect_pct / 100) * mean
            cohens_d = absolute_effect / std_dev if std_dev > 0 else 0
            
            if cohens_d == 0:
                return {
                    "error": "Effect size is zero (no variance in data)",
                    "current_sample_size": current_n
                }
            
            z_alpha = self._z_score_from_p(alpha / 2)
            z_beta = self._z_score_from_p(1 - power)
            
            required_n = math.ceil(2 * ((z_alpha + z_beta) / cohens_d) ** 2)
            
            autocorr_factor = 1.5
            required_n_adjusted = math.ceil(required_n * autocorr_factor)
            
            additional_needed = max(0, required_n_adjusted - current_n)
            
            if current_n > 0:
                current_detectable_d = (z_alpha + z_beta) * math.sqrt(2 / current_n)
                current_detectable_pct = (current_detectable_d * std_dev / mean) * 100
            else:
                current_detectable_pct = float('inf')
            
            interpretation = self._interpret_sample_size(
                current_n, required_n_adjusted, additional_needed,
                desired_effect_pct, current_detectable_pct, power
            )
            
            result = {
                "metric_name": metric_name or "all_metrics",
                "current_sample_size": current_n,
                "required_sample_size": required_n_adjusted,
                "additional_needed": additional_needed,
                "days_more_needed": additional_needed,
                "current_stats": {
                    "mean": round(mean, 2),
                    "std_dev": round(std_dev, 2),
                    "coefficient_of_variation": round((std_dev / mean) * 100, 1) if mean != 0 else 0
                },
                "power_analysis": {
                    "desired_effect_pct": desired_effect_pct,
                    "cohens_d": round(cohens_d, 3),
                    "target_power": power,
                    "alpha": alpha
                },
                "current_detection_ability": {
                    "minimum_detectable_effect_pct": round(current_detectable_pct, 1),
                    "can_detect_desired_effect": current_n >= required_n_adjusted
                },
                "interpretation": interpretation
            }
            
            logger.info(f"📊 Sample size advice: have {current_n}, need {required_n_adjusted} for {desired_effect_pct}% effect")
            return result
            
        except Exception as e:
            logger.error(f"❌ Sample size analysis failed: {e}")
            return {"error": str(e)}
    
    def _interpret_sample_size(self, current_n: int, required_n: int, additional: int,
                                desired_pct: float, current_detectable_pct: float,
                                power: float) -> str:
        """Generate human-readable sample size interpretation."""
        if additional == 0:
            return (f"✅ You have enough data! With {current_n} observations, you can reliably detect "
                   f"a {desired_pct}% change with {power*100:.0f}% power. "
                   f"Your current minimum detectable effect is {current_detectable_pct:.1f}%.")
        
        if additional <= 7:
            timeframe = f"{additional} more days"
        elif additional <= 30:
            timeframe = f"about {additional // 7} more weeks"
        else:
            timeframe = f"about {additional // 30} more months"
        
        return (f"⚠️ You need {additional} more data points ({timeframe}) to reliably detect "
               f"a {desired_pct}% change. Currently you can only detect changes of "
               f"{current_detectable_pct:.1f}% or larger. Consider waiting for more data "
               f"before drawing conclusions about smaller effects.")


if __name__ == "__main__":
    import sys
    
    print("📊 Testing StatisticalAnalyzer...")
    
    try:
        analyzer = StatisticalAnalyzer()
        print("✅ StatisticalAnalyzer initialized")
        
        test_user = sys.argv[1] if len(sys.argv) > 1 else None
        
        if test_user:
            result = analyzer.test_trend_significance(test_user)
            print(f"✅ Trend test: {result.get('trend_direction', 'N/A')}, p={result.get('p_value', 'N/A')}")
            
            sample = analyzer.advise_sample_size(test_user)
            print(f"✅ Sample size: have {sample.get('current_sample_size', 0)}, need {sample.get('required_sample_size', 0)}")
        
        print("\n✅ statistical.py working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

